"""Rebuild `_all_tests_combined.py` — one commented-out copy of the suite.

The combined file exists so the whole suite can be read, or pasted somewhere,
as a single artefact. It is generated, never edited: run

    .venv/bin/python tests/_regenerate_combined.py

after adding or changing a test, and commit the result alongside it.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "_all_tests_combined.py"

# Build order, which is also the order the project was written in.
FILES = ["world.py",
         "primitives/test_geometry.py",
         "forces/test_separation.py",
         "forces/test_alignment.py",
         "forces/test_cohesion.py",
         "simulator/test_step.py",
         "simulator/test_rollout.py",
         "dataset/test_generate.py",
         "dataset/test_dataset.py",
         "dataset/test_windows.py"]

RULE = "# " + "=" * 74


def main():
    chunks = []
    for name in FILES:
        body = (HERE / name).read_text().splitlines()
        commented = "\n".join(("# " + line).rstrip() for line in body)
        chunks.append(f"{RULE}\n# {name}\n{RULE}\n{commented}")
    OUT.write_text("\n\n".join(chunks) + "\n")
    print(f"wrote {OUT.name}: {len(FILES)} files, {len(OUT.read_text().splitlines())} lines")


if __name__ == "__main__":
    main()
