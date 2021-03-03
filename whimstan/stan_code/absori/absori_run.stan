data {

     // Input for absori
     vector[10] atomicnumber;
     real sigma[10, 26, 721];
     real ion[10, 26, 10];
     vector[721] energy_base;
     vector[721] spec;

     // Data input

}

parameters {

    // absori parameters
    real logn0;
    real logt4;

    // all other parameters

}

transformed parameter {
    real n0=exp(logn0*-7);
    real t4=exp(logt4);

}

model {

  logn0 ~ normal();
  logt4 ~ normal();

  // prior for other parameters



}
