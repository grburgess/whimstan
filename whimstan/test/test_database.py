from pathlib import Path

from whimstan import SpectrumFactory, Database




def test_database(whim_population):

    factory = SpectrumFactory(whim_population, whim_T=1e7, whim_n0=1e-7)

    factory.create_database("whim_db.h5")

    db = Database.read("whim_db.h5")

    sd = db.build_stan_data(use_host_gas=True, use_mw_gas=True, use_absori=True)

    db.catalog.plot_skymap()




    selection = db.catalog.z >1

    new_db = db.create_sub_selection(selection=selection)


    Path("whim_db.h5").unlink()
