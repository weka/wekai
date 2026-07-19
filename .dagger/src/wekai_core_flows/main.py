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
    "/wekai-core",
    "*.log",
    "*.out",
    "/results",
]


async def _calc_version(src: dagger.Directory) -> str:
    """Ported verbatim from wekai's .dagger/src/wekai_flows/main.py."""
    digest = await src.digest()
    sha = digest.split(":")[-1]
    version = f"v999.0.0-{sha[:12]}"
    return version


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
        COPY --from=<registry>:replay-<sha12> /replay.jsonl /wekai/replay.jsonl

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
        replay_image: str = "quay.io/weka.io/wekai-benchmark:replay-099a98c60fd7",
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
        version = await _calc_version(source)
        container = source.docker_build(
            platform=LINUX_AMD64,
            build_args=[BuildArg(name="REPLAY_IMAGE", value=replay_image)],
        )
        image_name = f"{registry}:{version}"
        await container.publish(image_name)
        return f"Published wekai-core image: {image_name}"
