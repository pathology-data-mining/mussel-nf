from pathlib import Path


def test_mussel_does_not_mutate_nextflow_output_file_lists():
    """Nextflow task output lists must not be sorted in place in channel operators."""
    mussel_nf = Path("modules/mussel.nf").read_text()

    assert "files_list.sort {" not in mussel_nf
    assert "files_list.toSorted {" in mussel_nf
