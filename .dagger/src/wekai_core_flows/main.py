from typing import Annotated

import dagger
from dagger import BuildArg, Ignore, dag, function, object_type

LINUX_AMD64 = dagger.Platform("linux/amd64")

# Mirrors wekai's own .gitignore so a publish digest isn't polluted by
# local build artifacts, IDE state, or the dagger module's own generated
# sdk/. Leading "/" anchors to the repo root — an unanchored "wekai-core"
# would also match chart/wekai-core (see wekai-core/.gitignore history).
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

CHART_DIR = "chart/wekai-core"
CHART_NAME = "wekai-core"


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
) -> tuple[str, str]:
    """Builds and publishes the wekai-core image from this repo's own
    Dockerfile. Shared by `publish` and `push_helm` so there is exactly one
    build/tag/publish code path — `push_helm` awaits this to completion
    before touching the chart, which is what makes "image pushed before
    chart push" hold by construction rather than by convention.

    Returns (image_name, version).
    """
    version = await _calc_version(source)
    container = source.docker_build(
        platform=LINUX_AMD64,
        build_args=[BuildArg(name="REPLAY_IMAGE", value=replay_image)],
    )
    image_name = f"{registry}:{version}"
    await container.publish(image_name)
    return image_name, version


@object_type
class WekaiCoreFlows:
    @function
    async def push_replay(
        self,
        replay: dagger.File,
        registry: str = "quay.io/weka.io/wekai-benchmark",
    ) -> str:
        """Publishes a router replay file as a minimal scratch image.

        Ported verbatim from wekai's .dagger push_replay: identical tag
        scheme (replay-<sha12 of content>) and default registry — the
        wekai-benchmark quay repo stays the historical home of replay
        artifacts, regardless of which project publishes or consumes them.
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
        registry: str = "quay.io/weka.io/wekai-core",
        replay_image: str = "quay.io/weka.io/wekai-benchmark:replay-24e7f15ba0ea",
    ) -> str:
        """Builds and publishes the wekai-core image from this repo's own Dockerfile.

        Uses Directory.docker_build() against the existing Dockerfile at the
        repo root — the Dockerfile is the single source of truth for the
        build; this function does not reimplement its steps. Tagged with
        the same version stamp scheme wekai uses (_calc_version:
        v999.0.0-<sha12> of the source directory's digest).

        Args:
            registry: Target image registry:repo (tag is appended).
            replay_image: Passed through to the Dockerfile's REPLAY_IMAGE
                build-arg, so a different embedded replay capture can be
                selected without editing the Dockerfile.
        """
        image_name, _version = await _publish_image(source, registry, replay_image)
        return f"Published wekai-core image: {image_name}"

    @function
    async def push_helm(
        self,
        source: Annotated[dagger.Directory, Ignore(SOURCE_IGNORE)],
        helm_username: dagger.Secret,
        helm_password: dagger.Secret,
        registry: str = "quay.io/weka.io/wekai-core",
        helm_registry: str = "quay.io/weka.io/helm",
        replay_image: str = "quay.io/weka.io/wekai-benchmark:replay-24e7f15ba0ea",
    ) -> str:
        """Publishes the wekai-core image, then packages and pushes
        chart/wekai-core to an OCI Helm registry with the chart's image
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
        """
        image_name, version = await _publish_image(source, registry, replay_image)

        chart = source.directory(CHART_DIR)
        registry_host = helm_registry.split("/")[0]  # e.g. "quay.io"

        packaged = (
            dag.container(platform=LINUX_AMD64)
            .from_("alpine:latest")
            .with_exec(["apk", "add", "--no-cache", "helm"])
            .with_directory("/chart", chart)
            # Version pinning is pure propagation (same pattern as wekai's
            # push_restricted): only Chart.yaml is stamped, and the deployment
            # template resolves the image tag via `imageTag | default
            # .Chart.AppVersion`. values.yaml imageTag stays "" — no hardcoded
            # version in the packaged chart. imageRepository IS synced to the
            # actual push destination so a custom registry never yields a
            # chart pointing at the default repo.
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
                        f"helm push /out/{CHART_NAME}-*.tgz oci://{helm_registry}"])
        ).stdout()

        return (
            f"Published wekai-core image: {image_name}\n"
            f"Published Helm chart: oci://{helm_registry}/{CHART_NAME}:{version} "
            f"(pinned to image {image_name})"
        )
