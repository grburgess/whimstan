vector powerlaw(vector ene, real K, real index) {

  real piv = 1.; // keV

  return K * pow(ene/piv, index);


}
