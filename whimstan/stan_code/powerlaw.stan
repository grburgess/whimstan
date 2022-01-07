
vector powerlaw_flux(vector ene, real index) {

  real dp2 = 2. + index;

  real inv_int_flux =  dp2 / ((ehi()^dp2) - (elo()^dp2));

  return inv_int_flux * erg2keV() * ene^index;

}
