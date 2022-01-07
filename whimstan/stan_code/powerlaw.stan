
vector powerlaw(vector ene, real K, real index) {

  real piv = 1.; // keV

  return K * pow(ene/piv, index);


}


vector powerlaw_flux(vector ene, real index, real a, real b) {

  real dp2 = 2 + index;

  real inv_int_flux =  inv((pow(b, dp2) - pow(a, dp2))) * dp2;


       
  return inv_int_flux * erg2keV() * pow(ene, index);


}
