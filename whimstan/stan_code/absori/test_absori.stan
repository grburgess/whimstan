functions {
#include absori.stan
}

data {

     real z;
     //real temp;
     real xi;
     int atomicnumber[10];
     real ion[10,26,10];
     real sigma[10,26,721];
     vector[721] spec;
     vector[10] abundance;

     vector[100] e_edges;
     matrix[10,26] sigma_interp[50*10,100];

     vector[100] observed;
}

transformed data{

    int num_e_edges=size(e_edges);
    //matrix[10,26] num;
    //vector[100] abs;
    //num = calc_num(spec, temp, xi, atomicnumber, sigma, ion);
    //for (i in 1:10){
    //    num[i] = abundance[i]*num[i];
    //}
    //abs = integrate_absori(z, 1e-5, num, abundance, sigma_interp, num_e_edges);

}

parameters {

    real logn0;
    real logt;

}

transformed parameters {

    real n0=pow(10,logn0);
    real temp=pow(10, logt);
    matrix[10,26] num;
    vector[100] abs;
    profile("calc_num") {
        num = calc_num(spec, temp, xi, atomicnumber, sigma, ion);
    }
    profile("mult_abundance"){
        for (i in 1:10){
            num[i] = abundance[i]*num[i];
            }
    }
    profile("integrate_absori") {
        abs = integrate_absori(z, n0, num, abundance, sigma_interp, num_e_edges);
    }
    //matrix[10,26] num = calc_num(spec, temp, xi, atomicnumber, sigma, ion);
    //vector[100] abs = integrate_absori(z, n0, num, abundance, sigma_interp, num_e_edges);

}

model {

    logn0 ~ normal(-4,1);
    logt ~ normal(5,1);

    observed ~ normal(abs, 0.1);

}
