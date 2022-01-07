
vector powerlaw_flux(vector ene, real index, real elo, real ehi) {

  real dp2 = 2. + index;

  real inv_int_flux =  inv((ehi^dp2) - (elo^dp2)) * dp2;

  return inv_int_flux * erg2keV() * ene^index;

}
