from typing import Annotated

import dagger
from dagger import BuildArg, Ignore, dag, function, object_type

LINUX_AMD64 = dagger.Platform("linux/amd64")

# Mirrors wekai's own .gitignore so a publish digest isn't polluted by
# local build artifacts, IDE state, or the dagger module's own generated
# sdk/. Leading "/" anchors to the repo root — an unanchored "wekai"
# would also match chart/wekai (see .gitignore history).
SOURCE_IGNORE = [
    "*/.git",
    "*/.DS_Store",
    ".dagger",
    "!.dagger/src/",
    ".idea",
    ".temp",
    "/bin",
    "/dist",
    "/wekai",
    "/wekai-core",
    "*.log",
    "*.out",
    "/results",
]

CHART_DIR = "chart/wekai"
CHART_NAME = "wekai"

# The router ships as a SECOND image and chart from the same commit. It is the
# same wekai binary; what differs is that the router image carries no embedded
# replay artifact. That artifact is multi-GB and exists so a benchmark pod can
# replay captured traffic without a volume — a router never opens it, and making
# every router replica pull it would be pure cost. Two Dockerfiles rather than a
# build arg, so a router image that accidentally includes it is impossible
# rather than merely unlikely.
ROUTER_DOCKERFILE = "Dockerfile.router"
ROUTER_CHART_DIR = "chart/router"
ROUTER_CHART_NAME = "wekai-router"


async def _calc_version(src: dagger.Directory) -> str:
    """Ported verbatim from wekai's .dagger/src/wekai_flows/main.py."""
    digest = await src.digest()
    sha = digest.split(":")[-1]
    version = f"v999.0.0-{sha[:12]}"
    return version


async def _publish_image(
    source: dagger.Directory,
    registry: str,
    replay_image: str,
    version: str = "",
) -> tuple[str, str]:
    """Builds and publishes the wekai image from this repo's own
    Dockerfile. Shared by `publish` and `push_helm` so there is exactly one
    build/tag/publish code path — `push_helm` awaits this to completion
    before touching the chart, which is what makes "image pushed before
    chart push" hold by construction rather than by convention.

    Returns (image_name, version).

    version: explicit version stamp (e.g. a semver tag from CI); when empty,
    falls back to the content-hash scheme (_calc_version's v999.0.0-<sha12>).
    """
    if not version:
        version = await _calc_version(source)
    container = source.docker_build(
        platform=LINUX_AMD64,
        build_args=[BuildArg(name="REPLAY_IMAGE", value=replay_image)],
    )
    image_name = f"{registry}:{version}"
    await container.publish(image_name)
    return image_name, version


async def _publish_router_image(
    source: dagger.Directory,
    registry: str,
    version: str = "",
) -> tuple[str, str]:
    """Builds and publishes the replay-less router image.

    Same source, same binary, different Dockerfile — see ROUTER_DOCKERFILE.

    Returns (image_name, version).
    """
    if not version:
        version = await _calc_version(source)
    container = source.docker_build(
        platform=LINUX_AMD64,
        dockerfile=ROUTER_DOCKERFILE,
    )
    image_name = f"{registry}:{version}"
    await container.publish(image_name)
    return image_name, version


async def _package_and_push_chart(
    chart: dagger.Directory,
    chart_name: str,
    registry: str,
    version: str,
    helm_registry: str,
    helm_username: dagger.Secret,
    helm_password: dagger.Secret,
) -> str:
    """Packages one chart pinned to one just-published image and pushes it.

    Version pinning is pure propagation: only Chart.yaml is stamped, and the
    deployment template resolves the tag via `imageTag | default
    .Chart.AppVersion`, so values.yaml never carries a hardcoded version.
    imageRepository IS synced to the real push destination, so publishing to a
    custom registry cannot yield a chart pointing at the default one.
    """
    registry_host = helm_registry.split("/")[0]
    packaged = (
        dag.container(platform=LINUX_AMD64)
        .from_("alpine:latest")
        .with_exec(["apk", "add", "--no-cache", "helm"])
        .with_directory("/chart", chart)
        .with_exec(["sed", "-i", f"s/^version:.*/version: {version}/", "/chart/Chart.yaml"])
        .with_exec(["sed", "-i", f's/^appVersion:.*/appVersion: "{version}"/', "/chart/Chart.yaml"])
        .with_exec(["sed", "-i", f"s|^imageRepository:.*|imageRepository: {registry}|", "/chart/values.yaml"])
        .with_exec(["helm", "package", "/chart", "--destination", "/out"])
    )
    await (
        packaged
        .with_secret_variable("HELM_USER", helm_username)
        .with_secret_variable("HELM_PASS", helm_password)
        .with_exec(["sh", "-c",
                    f"echo $HELM_PASS | helm registry login {registry_host} -u $HELM_USER --password-stdin && "
                    f"helm push /out/{chart_name}-*.tgz oci://{helm_registry}"])
    ).stdout()
    return f"oci://{helm_registry}/{chart_name}:{version}"


@object_type
class WekaiCoreFlows:
    @function
    async def push_replay(
        self,
        replay: dagger.File,
        registry: str = "quay.io/weka.io/wekai",
    ) -> str:
        """Publishes a router replay file as a minimal scratch image.

        Ported from wekai's .dagger push_replay: identical tag scheme
        (replay-<sha12 of content>). Replay artifacts live in the same
        wekai quay repo as the app image, distinguished by the replay-
        tag prefix.
        The image carries the replay JSONL at /replay.jsonl, so this repo's
        own Dockerfile embeds it via:
        COPY --link --from=<registry>:replay-<sha12> /replay.jsonl /wekai/replay.jsonl

        Args:
            replay: The local replay JSONL file to publish (mandatory).
        """
        sha_line = await (
            dag.container(platform=LINUX_AMD64)
            .from_("alpine:latest")
            .with_file("/replay.jsonl", replay)
            .with_exec(["sha256sum", "/replay.jsonl"])
            .stdout()
        )
        sha = sha_line.split()[0]
        tag = f"replay-{sha[:12]}"

        replay_image = dag.container(platform=LINUX_AMD64).with_file(
            "/replay.jsonl", replay
        )
        image_name = f"{registry}:{tag}"
        await replay_image.publish(image_name)

        return (
            f"Published replay image: {image_name}\n"
            f"Replay file path in image: /replay.jsonl\n"
            f"sha256: {sha}"
        )

    @function
    async def publish(
        self,
        source: Annotated[dagger.Directory, Ignore(SOURCE_IGNORE)],
        registry: str = "quay.io/weka.io/wekai",
        replay_image: str = "quay.io/weka.io/wekai:replay-24e7f15ba0ea",
        version: str = "",
    ) -> str:
        """Builds and publishes the wekai image from this repo's own Dockerfile.

        Uses Directory.docker_build() against the existing Dockerfile at the
        repo root — the Dockerfile is the single source of truth for the
        build; this function does not reimplement its steps. Tagged with
        the same version stamp scheme wekai uses (_calc_version:
        v999.0.0-<sha12> of the source directory's digest) unless an
        explicit version is given.

        Args:
            registry: Target image registry:repo (tag is appended).
            replay_image: Passed through to the Dockerfile's REPLAY_IMAGE
                build-arg, so a different embedded replay capture can be
                selected without editing the Dockerfile.
            version: Explicit version stamp (e.g. a semver tag from the
                release workflow). Empty = content-hash scheme.
        """
        image_name, _version = await _publish_image(source, registry, replay_image, version)
        return f"Published wekai image: {image_name}"

    @function
    async def push_helm(
        self,
        source: Annotated[dagger.Directory, Ignore(SOURCE_IGNORE)],
        helm_username: dagger.Secret,
        helm_password: dagger.Secret,
        registry: str = "quay.io/weka.io/wekai",
        helm_registry: str = "quay.io/weka.io/helm",
        replay_image: str = "quay.io/weka.io/wekai:replay-24e7f15ba0ea",
        version: str = "",
    ) -> str:
        """Publishes the wekai image, then packages and pushes
        chart/wekai to an OCI Helm registry with the chart's image
        reference pinned to that exact just-published image — a
        `helm install` of the pushed chart with zero further --set flags
        deploys exactly the image it was packaged with.

        Reuses the same image-publish path as `publish` (via the shared
        _publish_image helper — no duplicated build/tag/publish logic) and
        awaits it to completion before packaging the chart, so "image
        pushed before chart push" holds by construction.

        Args:
            helm_username: OCI Helm registry username.
            helm_password: OCI Helm registry password.
            registry: Image registry:repo the chart gets pinned to (same
                target `publish` would use).
            helm_registry: OCI Helm registry base, e.g. quay.io/weka.io/helm
                (matches wekai's retired chart-push flow's default).
            replay_image: Passed through to the Dockerfile's REPLAY_IMAGE
                build-arg.
            version: Explicit version stamp for BOTH the image tag and the
                chart version/appVersion (e.g. a semver tag from the release
                workflow). Empty = content-hash scheme (v999.0.0-<sha12>).
        """
        image_name, version = await _publish_image(source, registry, replay_image, version)

        chart_ref = await _package_and_push_chart(
            source.directory(CHART_DIR), CHART_NAME, registry, version,
            helm_registry, helm_username, helm_password,
        )

        return (
            f"Published wekai image: {image_name}\n"
            f"Published Helm chart: {chart_ref} (pinned to image {image_name})"
        )

    @function
    async def push_router_helm(
        self,
        source: Annotated[dagger.Directory, Ignore(SOURCE_IGNORE)],
        helm_username: dagger.Secret,
        helm_password: dagger.Secret,
        registry: str = "quay.io/weka.io/wekai-router",
        helm_registry: str = "quay.io/weka.io/helm",
        version: str = "",
    ) -> str:
        """Publishes the REPLAY-LESS router image, then packages and pushes
        chart/router pinned to it.

        The mirror of `push_helm` for the router half of a release. Both run
        from the same commit under the same semver, so a release yields two
        images and two charts that are known to agree: the benchmark pair
        carries the embedded replay artifact, the router pair does not.

        Args:
            helm_username: OCI Helm registry username.
            helm_password: OCI Helm registry password.
            registry: Image registry:repo for the router image.
            helm_registry: OCI Helm registry base.
            version: Explicit version stamp for both image tag and chart
                version/appVersion. Empty = content-hash scheme.
        """
        image_name, version = await _publish_router_image(source, registry, version)

        chart_ref = await _package_and_push_chart(
            source.directory(ROUTER_CHART_DIR), ROUTER_CHART_NAME, registry, version,
            helm_registry, helm_username, helm_password,
        )

        return (
            f"Published router image: {image_name}\n"
            f"Published Helm chart: {chart_ref} (pinned to image {image_name})"
        )
