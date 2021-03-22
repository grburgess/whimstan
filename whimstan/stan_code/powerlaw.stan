
vector powerlaw(vector ene, real K, real index) {

  real piv = 1.; // keV

  return K * pow(ene/piv, index);


}


vector powerlaw_eflux(vector ene, real K, real index, real a, real b) {

  real piv = 1.; // keV
  real intflux;
  
  
  if (index != -2.0){

    real dp2 = 2 + index;

    intflux = pow(piv, -index) * (pow(b, dp2) - pow(a, dp2)) / dp2;

	  }
  else {

    intflux = -piv*piv * log(a/b);

	  }
  
  return K * pow(ene/piv, index);


}
