#!/usr/bin/env python3
"""Validate active figure navigation identities without running scientific code.

Only root/current manuscript metadata are active. Dated ancestry is deliberately
not traversed. All referenced current files must resolve inside the supplied root.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_KEYS = {(d, str(i) if d == "main" else "S" + str(i))
                 for d, n in (("main", 8), ("supplementary", 7))
                 for i in range(1, n + 1)}


class Audit:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("root must be a directory")
        self.checks, self.errors = [], []
        self.identities = {}
        self.figures_checked = 0
        self.formats_checked = 0
        self.package_bindings_checked = 0

    def check(self, check_name, okay, **details):
        row = dict(check=check_name, passed=bool(okay), **details)
        self.checks.append(row)
        if not okay:
            self.errors.append(row)
        return bool(okay)

    def file(self, value, label):
        # Do not normalize '..' away before checking, or accept Windows drives.
        if not isinstance(value, str) or not value or "\\" in value or ":" in value:
            self.check(label, False, error="not a portable relative POSIX path", path=value)
            return None
        rel = PurePosixPath(value)
        if rel.is_absolute() or ".." in rel.parts or "." in value.split("/"):
            self.check(label, False, error="absolute or dot-segment path", path=value)
            return None
        target = self.root.joinpath(*rel.parts).resolve()
        try:
            target.relative_to(self.root)
        except ValueError:
            self.check(label, False, error="path or symlink escapes root", path=value)
            return None
        if not target.is_file():
            self.check(label, False, error="file missing", path=value)
            return None
        return target

    def actual(self, target):
        if target not in self.identities:
            h = hashlib.sha256()
            n = 0
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    n += len(block)
                    h.update(block)
            self.identities[target] = dict(bytes=n, sha256=h.hexdigest())
        return self.identities[target]

    def identity(self, record, label, path_override=None, require_bytes=True):
        if not isinstance(record, dict):
            self.check(label, False, error="identity must be an object")
            return None
        value = record.get("path") if path_override is None else path_override
        target = self.file(value, label + ".path")
        sha = record.get("sha256")
        valid_sha = isinstance(sha, str) and SHA_RE.fullmatch(sha) is not None
        self.check(label + ".sha256_format", valid_sha, path=value)
        size = record.get("bytes")
        valid_bytes = type(size) is int and size >= 0
        if require_bytes:
            self.check(label + ".bytes_format", valid_bytes, path=value)
        if target is not None:
            actual = self.actual(target)
            self.check(label + ".sha256", valid_sha and sha == actual["sha256"],
                       path=value, expected=sha, actual=actual["sha256"])
            if require_bytes or "bytes" in record:
                self.check(label + ".bytes", valid_bytes and size == actual["bytes"],
                           path=value, expected=size, actual=actual["bytes"])
        return target

    def document(self, value, label):
        target = self.file(value, label + ".path")
        if target is None:
            return None
        try:
            d = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(d, dict):
                raise ValueError("top-level JSON must be an object")
        except (OSError, UnicodeError, ValueError) as exc:
            self.check(label + ".parse", False, path=value, error=str(exc))
            return None
        self.check(label + ".parse", True, path=value)
        return d

    def figures(self, index, specs, numbering):
        figures = index.get("figures", [])
        if not isinstance(figures, list):
            self.check("index.figures", False, error="figures must be a list")
            return
        self.check("index.figure_count", len(figures) == 15, actual=len(figures), expected=15)
        keyed, by_label, format_count, package_bindings = {}, {}, 0, 0
        for pos, fig in enumerate(figures):
            prefix = "index.figures[%d]" % pos
            if not isinstance(fig, dict):
                self.check(prefix, False, error="figure must be an object")
                continue
            key = (fig.get("document"), str(fig.get("figure")))
            self.check(prefix + ".unique_number", key not in keyed, number=list(key))
            keyed[key] = fig
            label = fig.get("label")
            self.check(prefix + ".unique_label", isinstance(label, str) and label not in by_label,
                       label=label)
            by_label[label] = fig
            package_file = fig.get("package_file")
            if package_file is not None:
                package_bindings += 1
                self.identity(fig, prefix + ".package_file", path_override=package_file)
                main_file = package_file
            else:
                main_file = "manuscript/" + fig.get("file", "")
                self.identity(fig, prefix + ".main_file", path_override=main_file)
            formats = fig.get("formats", [])
            self.check(prefix + ".format_count", isinstance(formats, list) and len(formats) == 3,
                       expected=3, actual=len(formats) if isinstance(formats, list) else None)
            if not isinstance(formats, list):
                continue
            suffixes = []
            for j, fmt in enumerate(formats):
                if not isinstance(fmt, dict):
                    self.check(prefix + ".formats[%d]" % j, False, error="not an object")
                    continue
                original = fmt.get("path", "")
                # A package explicitly binds the unchanged PDF; other formats
                # must have the same stem and resolve beside that bound PDF.
                suffix = PurePosixPath(original).suffix
                suffixes.append(suffix)
                bound = str(PurePosixPath(main_file).with_suffix(suffix)) if package_file else original
                self.check(prefix + ".formats[%d].stem" % j,
                           PurePosixPath(original).stem == PurePosixPath(main_file).stem,
                           path=original, bound_path=bound)
                self.identity(fmt, prefix + ".formats[%d]" % j, path_override=bound)
                format_count += 1
            self.check(prefix + ".format_suffixes", sorted(suffixes) == [".pdf", ".png", ".svg"],
                       actual=suffixes)
            pdf_records = [x for x in formats if isinstance(x, dict)
                           and PurePosixPath(x.get("path", "")).suffix == ".pdf"]
            if len(pdf_records) == 1:
                pdf = pdf_records[0]
                self.check(prefix + ".pdf_identity_matches_figure",
                           pdf.get("bytes") == fig.get("bytes") and pdf.get("sha256") == fig.get("sha256"))
        self.check("index.figure_numbers", set(keyed) == EXPECTED_KEYS,
                   actual=sorted([list(x) for x in keyed], key=str))
        self.check("index.total_formats", format_count == 45, actual=format_count, expected=45)
        self.figures_checked = len(figures)
        self.formats_checked = format_count
        self.package_bindings_checked = package_bindings
        self.check("index.package_binding_count", package_bindings in (0, 15),
                   actual=package_bindings, expected="0 outer or 15 extracted-package bindings")

        mappings = numbering.get("mappings", [])
        self.check("numbering.mapping_count", isinstance(mappings, list) and len(mappings) == 15)
        seen = set()
        if isinstance(mappings, list):
            for pos, mapping in enumerate(mappings):
                p = "numbering.mappings[%d]" % pos
                if not isinstance(mapping, dict):
                    self.check(p, False, error="not an object")
                    continue
                key = (mapping.get("document"), str(mapping.get("figure")))
                self.check(p + ".unique_number", key not in seen)
                seen.add(key)
                reference = keyed.get(key, {})
                for name in ("label", "file", "sha256"):
                    self.check(p + "." + name, key in keyed and mapping.get(name) == reference.get(name),
                               expected=reference.get(name), actual=mapping.get(name))
        self.check("numbering.figure_numbers", seen == EXPECTED_KEYS)

        spec_figures = specs.get("figures", [])
        self.check("specs.figure_count", isinstance(spec_figures, list) and len(spec_figures) == 15)
        seen_specs = set()
        if isinstance(spec_figures, list):
            for pos, spec in enumerate(spec_figures):
                p = "specs.figures[%d]" % pos
                if not isinstance(spec, dict):
                    self.check(p, False, error="not an object")
                    continue
                key = (spec.get("main_or_supp", spec.get("document")), str(spec.get("figure")))
                self.check(p + ".unique_number", key not in seen_specs)
                seen_specs.add(key)
                reference = keyed.get(key, {})
                self.check(p + ".label", key in keyed and spec.get("label") == reference.get("label"))
                self.check(p + ".formats", key in keyed and spec.get("formats") == reference.get("formats"))
                self.check(p + ".caption_plain", key in keyed and spec.get("caption_plain") == reference.get("caption_plain"))
        self.check("specs.figure_numbers", seen_specs == EXPECTED_KEYS)

    def run(self):
        claim_paths = [p for p in ("CLAIM_EVIDENCE_MAP.json", "manuscript/CLAIM_EVIDENCE_MAP.json")
                       if (self.root / p).is_file()]
        self.check("current_claim_map_present", bool(claim_paths))
        docs = []
        for rel in claim_paths:
            claim = self.document(rel, rel)
            if claim is None:
                continue
            nav = claim.get("current_numbering_navigation")
            if not isinstance(nav, dict):
                self.check(rel + ".current_numbering_navigation", False, error="missing object")
                continue
            records = nav.get("files")
            self.check(rel + ".navigation_files", isinstance(records, list) and len(records) == 2)
            if isinstance(records, list):
                names = set()
                for pos, record in enumerate(records):
                    self.identity(record, rel + ".navigation.files[%d]" % pos)
                    if isinstance(record, dict):
                        names.add(PurePosixPath(record.get("path", "")).name)
                self.check(rel + ".navigation_basenames", names == {
                    "CURRENT_FIGURE_NUMBERING.en.md", "CURRENT_FIGURE_NUMBERING.json"})
            index_record = nav.get("authoritative_index", {})
            self.identity(index_record, rel + ".navigation.authoritative_index")
            docs.append(index_record.get("path"))
        numbering = self.document("CURRENT_FIGURE_NUMBERING.json", "current_numbering")
        if numbering is None:
            return self.result()
        for kind in ("index", "specs"):
            record = dict(path=numbering.get("authoritative_" + kind),
                          sha256=numbering.get("authoritative_" + kind + "_sha256"))
            self.identity(record, "numbering.authoritative_" + kind, require_bytes=False)
        index = self.document(numbering.get("authoritative_index"), "numbering.index")
        specs = self.document(numbering.get("authoritative_specs"), "numbering.specs")
        for rel in docs:
            # A root-level copy may coexist with a canonical current-metadata
            # path, but both must bind byte-identical active index contents.
            target = self.file(rel, "claim_numbering_index_consistency.path")
            canonical = self.file(numbering.get("authoritative_index"), "claim_numbering_index_consistency.canonical")
            if target is not None and canonical is not None:
                self.check("claim_numbering_index_consistency", self.actual(target) == self.actual(canonical),
                           claim_index=rel, numbering_index=numbering.get("authoritative_index"))
        if index is not None and specs is not None:
            self.figures(index, specs, numbering)
        return self.result()

    def result(self):
        return dict(schema="current-navigation-validation/1.0",
                    created_at_utc=datetime.now(timezone.utc).isoformat(), root=str(self.root),
                    status="PASS_CURRENT_NAVIGATION" if not self.errors else "FAIL_CURRENT_NAVIGATION",
                    scope="Active root/manuscript claim navigation, current numbering/index/specs, and 15 figures/45 formats; no historical ancestry or scientific execution.",
                    checks_run=len(self.checks), checks_failed=len(self.errors),
                    figures=self.figures_checked, formats=self.formats_checked,
                    package_bindings=self.package_bindings_checked,
                    unique_files_hashed=len(self.identities), failures=self.errors, checks=self.checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        parser.error("--output must be outside --root; immutable input directories are never modified")
    try:
        result = Audit(root).run()
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        result = dict(schema="current-navigation-validation/1.0", status="FAIL_CURRENT_NAVIGATION",
                      root=str(root), error=type(exc).__name__ + ": " + str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "checks_run", "checks_failed", "figures", "formats", "unique_files_hashed")
                      if k in result}, ensure_ascii=False))
    return 0 if result["status"] == "PASS_CURRENT_NAVIGATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
