def test_scgeo_public_imports():
    import scgeo
    import scgeo.pl
    import scgeo.tl

    assert callable(scgeo.tl.shift)
    assert callable(scgeo.tl.mixscore)
    assert callable(scgeo.tl.distribution_test)
    assert callable(scgeo.tl.velocity_shift_alignment)
    assert callable(scgeo.pl.recovery_compass)
    assert callable(scgeo.pl.composition_drift)
