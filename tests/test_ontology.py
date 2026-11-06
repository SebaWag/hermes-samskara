"""Offline tests for the ontology: subsumption, paths, materialisation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samskara_core.ontology import (  # noqa: E402
    Ontology,
    materialise_nt,
    parse_nt_line,
)

NS = "https://wsolutionsai.com/ns/shiva/"
TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def test_superclasses_are_transitive():
    o = Ontology()
    assert {"Evento", "Ocurrencia", "Entidad"} <= o.superclasses("Incidente")
    assert "Incidente" not in o.superclasses("Incidente")


def test_subclasses_list_specialisations():
    o = Ontology()
    assert {"Incidente", "Reunion", "Despliegue"} == o.subclasses("Evento")


def test_is_a_walks_the_dag():
    o = Ontology()
    assert o.is_a("Incidente", "Entidad")
    assert not o.is_a("Entidad", "Incidente")


def test_predicate_hierarchy():
    o = Ontology()
    assert {"dependeDe", "relacionadoCon"} == o.superpredicates("usaTecnologia")
    path = o.predicate_path("dependeDe")
    assert "usaTecnologia" in path and "desplegadoEn" in path and path.startswith("(")


def test_parse_nt_normalises_brackets():
    s, p, o = parse_nt_line(f"<{NS}a> <{TYPE}> <{NS}Incidente> .")
    assert s == f"{NS}a" and p == TYPE and o == f"<{NS}Incidente>"


def test_parse_nt_keeps_literals_quoted():
    line = f'<{NS}a> <http://www.w3.org/2000/01/rdf-schema#label> "Caída" .'
    _, _, o = parse_nt_line(line)
    assert o.startswith('"')


def test_parse_nt_rejects_garbage():
    assert parse_nt_line("not a triple") is None
    assert parse_nt_line("") is None
    assert parse_nt_line("# comment") is None


def test_materialise_adds_ancestors():
    nt = f"<{NS}inc> <{TYPE}> <{NS}Incidente> ."
    out, added = materialise_nt(nt)
    assert added == 3  # Evento, Ocurrencia, Entidad
    for ancestor in ("Evento", "Ocurrencia", "Entidad"):
        assert f"<{NS}{ancestor}>" in out


def test_materialise_is_idempotent_on_unknown_types():
    nt = f"<{NS}x> <{TYPE}> <{NS}NoExiste> ."
    out, added = materialise_nt(nt)
    assert added == 0 and out == nt


def test_materialise_ignores_non_type_predicates():
    nt = f'<{NS}a> <http://www.w3.org/2000/01/rdf-schema#label> "x" .'
    _, added = materialise_nt(nt)
    assert added == 0


def test_every_declared_parent_exists_or_is_a_root():
    o = Ontology()
    for child, parents in o.classes.items():
        for parent in parents:
            assert parent in o.classes or parent == "Entidad", f"{child} -> {parent}"
