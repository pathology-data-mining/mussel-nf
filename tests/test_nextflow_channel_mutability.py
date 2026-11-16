from pathlib import Path


def test_mussel_does_not_mutate_nextflow_output_file_lists():
    """Nextflow task output lists must not be sorted in place in channel operators."""
    mussel_nf = Path("modules/mussel.nf").read_text()

    assert "files_list.sort {" not in mussel_nf


def test_batch_outputs_are_paired_with_slides_by_slide_id():
    """Batch outputs pair with their slide by the slide ID in the file name, never by position.

    A Mussel >= 1.6.0 batch task writes nothing for a slide that failed or had no
    tiles, so sorted lists of metas and files no longer line up.
    """
    mussel_nf = Path("modules/mussel.nf").read_text()

    assert "sorted_meta" not in mussel_nf and "sorted_files" not in mussel_nf
    assert "def bySlide(" in mussel_nf and "def slideIdOf(" in mussel_nf
