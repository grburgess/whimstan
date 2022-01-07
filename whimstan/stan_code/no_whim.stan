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
  //array[N_grbs] matrix[N_chan, N_ene] rsp;

  matrix[N_chan, N_ene] rmf;
  array[N_grbs] vector[N_ene] arf;
  vector[N_grbs] z; //redshift
  vector[N_grbs] nH_mw;
  vector[N_grbs] exposure_ratio;
  array[N_grbs] vector[N_ene] precomputed_absorp;
  array[N_grbs] vector[N_ene] host_precomputed_absorp;
  array[N_grbs] vector[N_ene] ene_avg;
  array[N_grbs] vector[N_ene] ene_width;
  array[N_grbs] int n_chans_used;
  array[N_grbs] vector[N_chan] counts;
  array[N_grbs] vector[N_chan] bkg;
  array[N_grbs,N_chan] int mask;
  vector[N_grbs] exposure;
}


transformed data{
  array[N_grbs] int all_N;
  array[N_grbs] vector[N_chan] log_fact_obs;
  array[N_grbs] vector[N_chan] log_fact_bkg;
  array[N_grbs] vector[N_chan] o_plus_b;
  array[N_grbs] vector[N_chan] alpha_bkg_factor;



  int grainsize = 1;


  //mw abs is fixed
  array[N_grbs] vector[N_ene] mw_abs;

  for (n in 1:N_grbs){
    mw_abs[n] = absorption(nH_mw[n], precomputed_absorp[n]);
  }


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

  //vector[N_grbs] index;

  // vector<lower=0>[N_grbs] nH_mw;

}


transformed parameters{
  vector[N_grbs] index;
  vector[N_grbs] log_K; // log eflux
  vector[N_grbs] K;
  vector[N_grbs] log_nH_host;
  //  vector[N_grbs] nH_host;
  vector[N_grbs] nH_host_norm;
  real log_K_mu = log_K_mu_raw - 9;



  // non centered parameterizartion

  log_nH_host = log_nH_host_mu_raw + log_nH_host_raw * log_nH_host_sigma;

  index = index_mu + index_raw * index_sigma;

  log_K = log_K_mu + log_K_raw * log_K_sigma;

  K =  pow(10.,log_K);

  nH_host_norm = pow(10.,log_nH_host);

}


model{



  host_alpha ~ normal(-1,2);

  index_raw ~ std_normal();
  log_K_raw ~ std_normal();

  //log_nH_host_raw ~ skew_normal(0,1, host_alpha);

  log_nH_host_raw ~ normal(0,1);

  target += normal_lcdf(host_alpha * log_nH_host_raw | 0, 1);

  log_nH_host_mu_raw ~ std_normal();
  log_nH_host_sigma ~ std_normal();

  log_K_mu_raw ~ std_normal();
  log_K_sigma ~ std_normal();
  //index ~ normal(-2, 0.5);

  index_mu ~ normal(-2, .1);
  index_sigma ~ std_normal();


  target += reduce_sum(pll_no_whim,
                       all_N,
                       grainsize,
                       N_ene, N_chan,
                       host_precomputed_absorp,
                       precomputed_absorp,
                       ene_avg,
                       ene_width,
                       mask,
                       n_chans_used,
                       K,
                       index,
                       nH_host_norm,
                       mw_abs,
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
                       alpha_bkg_factor );

}

generated quantities {

  real log_nH_host_mu = log_nH_host_mu_raw + 22;


}
