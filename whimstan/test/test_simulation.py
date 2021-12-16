from whimstan.simulation import SpectrumFactory


def test_whim_simulation(whim_population):

    factory = SpectrumFactory(whim_population, whim_T=1E7, whim_n0=1E-7)

    factory.create_database("whim_db.h5")



def test_no_whim_simulation(population):

    factory = SpectrumFactory(population)

    factory.create_database("no_whim_db.h5")