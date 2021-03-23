
vector powerlaw(vector ene, real K, real index) {

  real piv = 1.; // keV

  return K * pow(ene/piv, index);


}


vector powerlaw_flux(vector ene, real K, real index) {


  real norm;
  real intflux;
  real erg2keV = 6.24151e8;	

  if (index != -2.0){

          real dp2 = 2 + index;

         intflux =  (pow(b, dp2) - pow(a, dp2)) / dp2;
     else{

         intflux = - log(a/b);
       }

    norm = (K / (intflux) ) * erg2keV
       
  return norm * pow(ene, index);


}
