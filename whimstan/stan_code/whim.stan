functions {
#include constants.stanfunctions
#include absori.stanfunctions
#include tbabs.stanfunctions
#include powerlaw.stanfunctions
#include cstat.stanfunctions
#include partial_log_like_optimized.stanfunctions
}

data{

  int N_grbs;
  int N_ene;
  int N_chan;

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

  // distributions
  real t_whim_lower;
  real t_whim_upper;
  real t_whim_mu;
  real t_whim_sigma;

  real K_offset;
  real nh_host_offset;

  real host_alpha_mu;
  real host_alpha_sigma;


}


transformed data{
  int all_N[N_grbs];

  array[N_grbs] vector[N_chan] log_fact_obs;
  array[N_grbs] vector[N_chan] log_fact_bkg;
  array[N_grbs] vector[N_chan] o_plus_b;
  array[N_grbs] vector[N_chan] alpha_bkg_factor;
  array[N_grbs] vector[N_chan] zero_mask;

  real log_t4_raw_lower = log_t_whim_lower -6;
  real log_t4_raw_upper = log_t_whim_upper -6;


  int num_energy_base=size(sigma[1,1]);
  int num_atomicnumber=size(atomicnumber);
  int max_atomicnumber=max(atomicnumber);

  int num_size = num_atomicnumber * max_atomicnumber;


  vector[num_size] zero_matrix = rep_vector(0., num_size);


  vector[max_atomicnumber] zero_vector  = rep_vector(0., max_atomicnumber);

  array[num_atomicnumber] vector[max_atomicnumber] precalc_intgral;

  int grainsize = 1;

  // This version makes the array for sigma
  // as one matrix per GRB


  array[N_grbs] matrix [N_ene, num_atomicnumber * max_atomicnumber] sum_sigma_interp_vec;

  // fill the array



  for (i in 1:N_grbs) {


    for (j in 1:N_ene) {
      for (k in 1:num_atomicnumber) {
        for (l in 1:max_atomicnumber) {

          sum_sigma_interp_vec[i][j, (k-1)*max_atomicnumber +l ] = sum_sigma_interp[i,j,k,l];

        }
      }
    }
  }






  // precalc for num

  for (i in 1:num_atomicnumber){

    int Ne = atomicnumber[i];

    for (j in 1:Ne){
      precalc_intgral[i][j] = 0.0;
      for (k in 1:num_energy_base){
        precalc_intgral[i][j] += sigma[i,j,k]*spec[k];
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

    log_fact_obs[n] = logfactorial(counts[n]);
    log_fact_bkg[n] = logfactorial(bkg[n]);

    o_plus_b[n] = counts[n] + bkg[n];

    alpha_bkg_factor[n] = 4 * (exposure_ratio[n] + square(exposure_ratio[n])) * bkg[n];

    for (m in 1:N_chan){

      if (bkg[n][m]>0){
        zero_mask[n][m] = 0;

      }

      else {

        zero_mask[n][m] = 1;

      }
    }

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

  real<lower=t_whim_lower, upper=t_whim_upper> log_t_whim;
  real<lower=t_whim_lower, upper=t_whim_upper> log_t4_whim_raw;
}


transformed parameters{
  vector[N_grbs] index;
  vector[N_grbs] log_K; // log eflux
  vector[N_grbs] K;
  vector[N_grbs] log_nH_host;

  vector[N_grbs] nH_host_norm;

  real log_n0_whim = log_n0_whim_raw -7;


  real log_K_mu = log_K_mu_raw + K_offset;
  real log_nH_host_mu_tmp = log_nH_host_mu_raw + nh_host_offset;

  // absori
  real n0_whim = pow(10, log_n0_whim);

  vector[num_size] num;

  //  real t_whim=pow(10,log_t_whim);


  num = calc_num_vec(pow(10., log_t4_whim_raw +2),
                     xi,
                     atomicnumber,
                     ion,
                     zero_matrix,
                     zero_vector,
                     precalc_intgral,
                     num_energy_base,
                     num_atomicnumber,
                     max_atomicnumber,
                     num_size
                     );



  for (i in 1:10){

    num[(i-1) * max_atomicnumber  +1 : i*max_atomicnumber] = abundance[i]*num[(i-1) * max_atomicnumber  +1 : i*max_atomicnumber];

  }

  // non centered parameterizartion

  log_nH_host = log_nH_host_mu_tmp + log_nH_host_raw * log_nH_host_sigma;

  index = index_mu + index_raw * index_sigma;

  log_K = log_K_mu + log_K_raw * log_K_sigma;

  K =  pow(10.,log_K);

  nH_host_norm = pow(10.,log_nH_host);



}


model{


  host_alpha ~ normal(host_alpha_mu, host_alpha_sigma);

  log_nH_host_mu_raw ~ std_normal();
  log_nH_host_sigma ~ normal(0.5, 0.5);

  log_nH_host_raw ~ std_normal();

  target += normal_lcdf(host_alpha * log_nH_host_raw | 0, 1);


  log_K_mu_raw ~ std_normal();
  log_K_sigma ~ normal(0.5, 0.5);


  index_mu ~ normal(-2, .2);
  index_sigma ~ normal(0.5, 0.5);

  index_raw ~ std_normal();
  log_K_raw ~ std_normal();


  //absori

  log_n0_whim_raw ~ normal(0, 1);

  // log_t_whim ~ normal(t_whim_mu, t_whim_sigma);
  log_t4_whim_raw ~ std_normal();


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
                       n0_whim,
                       num,
                       sum_sigma_interp_vec,
                       nH_host_norm,
                       host_precomputed_absorp,
                       rmf,
                       arf,
                       exposure,
                       exposure_ratio,
                       counts,
                       bkg,
                       log_fact_obs,
                       log_fact_bkg,
                       o_plus_b,
                       alpha_bkg_factor,
                       zero_mask
                       );



}

generated quantities {

  real log_nH_host_mu = log_nH_host_mu_tmp + 22;
  real log_t4 = log_t4_whim_raw + 2;
  real log_t_whim = log_t4 + 4;
  real t_whim = pow(10, log_t_whim);
  
}
