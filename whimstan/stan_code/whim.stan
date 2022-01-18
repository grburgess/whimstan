functions {
#include constants.stan
#include absori.stan
#include tbabs.stan
#include powerlaw.stan
#include cstat.stan
#include partial_log_like_optimized.stan
}

data{

  int N_grbs;
  int N_ene;
  int N_chan;
  //array [N_grbs] matrix[N_chan, N_ene] rsp;
  matrix[N_chan, N_ene] rmf;
  array[N_grbs] vector[N_ene] arf;
  vector[N_grbs] z; //redshift
  vector[N_grbs] nH_mw;
  array[N_grbs] vector[N_ene] precomputed_absorp;
  array[N_grbs] vector[N_ene] host_precomputed_absorp;
  vector[N_grbs] exposure_ratio;
  array[N_grbs] vector[N_ene] ene_avg;
  array[N_grbs] vector[N_ene] ene_width;
  array[N_grbs] int n_chans_used;
  array[N_grbs] vector[N_chan] counts;
  array[N_grbs] vector[N_chan] bkg;
  array[N_grbs,N_chan] int mask;
  vector[N_grbs] exposure;

  // absori input
  real xi;
  array[10] int atomicnumber;
  array[10,26,10] real ion;
  array[10,26,721] real sigma;
  vector[721] spec;
  vector[10] abundance;
  array[N_grbs, N_ene] matrix[10,26] sum_sigma_interp;
}


transformed data{
  int all_N[N_grbs];

  array[N_grbs] vector[N_chan] log_fact_obs;
  array[N_grbs] vector[N_chan] log_fact_bkg;
  array[N_grbs] vector[N_chan] o_plus_b;
  array[N_grbs] vector[N_chan] alpha_bkg_factor;

  int num_energy_base=size(sigma[1,1]);
  int num_atomicnumber=size(atomicnumber);
  int max_atomicnumber=max(atomicnumber);

  int num_size = num_atomicnumber * max_atomicnumber;

  // int N_shards = 4;
  // int shard_size = N_grbs/N_shards;

  // matrix[num_atomicnumber, max_atomicnumber] zero_matrix = rep_matrix(0., num_atomicnumber, max_atomicnumber);

  vector[num_size] zero_matrix = rep_vector(0., num_size);


  vector[max_atomicnumber] zero_vector  = rep_vector(0., max_atomicnumber);

  array[num_atomicnumber] vector[max_atomicnumber] precalc_intgral;

  int grainsize = 1;

  array[N_grbs , N_ene * num_atomicnumber * max_atomicnumber] real sum_sigma_interp_vec;
  array[N_grbs , 0] int x_i; // dummy
  array[N_grbs] vector[0] theta;
  // fill the array
  for (i in 1:N_grbs) {

    x_i[i] = {0};


    for (j in 1:N_ene) {
      for (k in 1:num_atomicnumber) {
        for (l in 1:max_atomicnumber) {

          sum_sigma_interp_vec[i][(j-1)*num_atomicnumber*max_atomicnumber + (k-1)*max_atomicnumber +l ] = sum_sigma_interp[i,j,k,l];

        }
      }
    }
  }


  // precalc for num
  profile("num_pre") {

    for (i in 1:num_atomicnumber){

      int Ne = atomicnumber[i];

      for (j in 1:Ne){
        precalc_intgral[i][j] = 0.0;
        for (k in 1:num_energy_base){
          precalc_intgral[i][j] += sigma[i,j,k]*spec[k];
        }
      }
    }

  }
  //mw abs is fixed
  array[N_grbs] vector[N_ene] mw_abs;

  for (n in 1:N_grbs){
    mw_abs[n] = absorption(nH_mw[n], precomputed_absorp[n]);
  }

  // precalculation of energy bounds

  for (n in 1:N_grbs) {

    all_N[n] = n;

  }

  // now do some static calculations for CSTAT


  for (n in 1:N_grbs) {

    for (m in 1:N_chan) {

      log_fact_obs[n,m] = logfactorial(counts[n,m]);


      if (bkg[n,m] >0) {

        log_fact_bkg[n,m] = logfactorial(bkg[n,m]);

      }

    }

    o_plus_b[n] = counts[n] + bkg[n];

    alpha_bkg_factor[n] = 4 * (exposure_ratio[n] + square(exposure_ratio[n])) * bkg[n];


  }





}

parameters{

  //vector<upper=0>[N_grbs] index;
  real log_nH_host_mu_raw;
  real<upper=0> host_alpha; // skew normal paramter

  real<lower=0> log_nH_host_sigma;
  vector[N_grbs] log_nH_host_raw;

  real index_mu;
  real<lower=0> index_sigma;
  vector[N_grbs] index_raw;


  real log_K_mu_raw;
  real<lower=0> log_K_sigma;


  vector[N_grbs] log_K_raw; // raw energy flux norm

  // absori parameter
  real log_n0_whim_raw;
  real<upper=8> log_t_whim;
}


transformed parameters{
  vector[N_grbs] index;
  vector[N_grbs] log_K; // log eflux
  vector[N_grbs] K;
  vector[N_grbs] log_nH_host;
  //  vector[N_grbs] nH_host;
  vector[N_grbs] nH_host_norm;

  real log_n0_whim = log_n0_whim_raw -7;


  real log_K_mu = log_K_mu_raw - 9;

  // absori
  real n0_whim = pow(10, log_n0_whim);

  // free temp
  //  matrix[10,26] num;

  vector[num_size] num;

  real t_whim=pow(10,log_t_whim);

  //array[N_grbs] vector[N_ene] whim_abs;

  vector[N_grbs * N_ene] whim_abs;

  profile("num") {

    num = calc_num_vec(//spec,
                       t_whim,
                       xi,
                       atomicnumber,
                       //sigma,
                       ion,
                       zero_matrix,
                       zero_vector,
                       precalc_intgral,
                       num_energy_base,
                       num_atomicnumber,
                       max_atomicnumber,
		       num_size
		       );

  }

  for (i in 1:10){

    num[(i-1) * max_atomicnumber  +1 : i*max_atomicnumber] = abundance[i]*num[(i-1) * max_atomicnumber  +1 : i*max_atomicnumber];

  }

  profile("whim_abs") {

    // for (n in 1:N_grbs) {

    //   whim_abs[n] = exp(integrate_absori_precalc(sum_sigma_interp[n], num, N_ene)* n0_whim);
    // }


    whim_abs =  exp(-map_rect(integrate_absori_vec3,num,theta,sum_sigma_interp_vec, x_i) * n0_whim);



  }


  // non centered parameterizartion

  log_nH_host = log_nH_host_mu_raw + log_nH_host_raw * log_nH_host_sigma;

  index = index_mu + index_raw * index_sigma;

  log_K = log_K_mu + log_K_raw * log_K_sigma;

  K =  pow(10.,log_K);

  nH_host_norm = pow(10.,log_nH_host);



}


model{


  host_alpha ~ normal(-2,1);

  index_raw ~ std_normal();
  log_K_raw ~ std_normal();
  log_nH_host_raw ~ std_normal();

  //log_nH_host_mu_raw ~ skew_normal(0,1, host_alpha);

  log_nH_host_raw ~ normal(0,1);

  target += normal_lcdf(host_alpha * log_nH_host_raw | 0, 1);


  log_nH_host_sigma ~ std_normal();

  log_K_mu_raw ~ std_normal();
  log_K_sigma ~ std_normal();


  index_mu ~ normal(-2, .1);
  index_sigma ~ std_normal();


  //absori

  log_n0_whim_raw ~ normal(0, 2);

  log_t_whim ~ normal(6, 2);

  profile("loglike") {

    target += reduce_sum(pll_whim,
                         all_N,
                         grainsize,
                         N_ene,
                         N_chan,
                         ene_avg,
                         ene_width,
                         mask,
                         n_chans_used,
                         mw_abs,
                         K,
                         index,
                         //                       n0_whim,
                         //num,
                         //sum_sigma_interp,
                         whim_abs,
                         nH_host_norm,
                         host_precomputed_absorp,

                         //rsp,
                         rmf,
                         arf,

                         exposure,
                         exposure_ratio,
                         counts,
                         bkg,
                         log_fact_obs,
                         log_fact_bkg,
                         o_plus_b,
                         alpha_bkg_factor);

  }

}

generated quantities {

  real log_nH_host_mu = log_nH_host_mu_raw + 22;

}
