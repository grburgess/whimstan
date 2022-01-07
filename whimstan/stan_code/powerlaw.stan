
vector powerlaw_flux(vector ene, real index, real a, real b) {

  real dp2 = 2 + index;

  real inv_int_flux =  inv(b^dp2 - a^dp2) * dp2;

  return inv_int_flux * erg2keV() * ene^index;


}
