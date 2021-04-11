functions {
#include absori.stan 
#include tbabs.stan
#include powerlaw.stan
#include cstat.stan
#include partial_log_like.stan
}

data{

  int N_grbs;
  int N_ene;
  int N_chan;
  matrix[N_chan, N_ene] rsp[N_grbs];
  vector[N_grbs] z; //redshift
  vector[N_grbs] nH_mw;
  vector[N_grbs] exposure_ratio;
  vector[N_ene] precomputed_absorp[N_grbs];
  vector[N_ene] host_precomputed_absorp[N_grbs];
  vector[N_ene] ene_avg[N_grbs];
  vector[N_ene] ene_width[N_grbs];
  int n_chans_used[N_grbs];
  vector[N_chan] counts[N_grbs];
  vector[N_chan] bkg[N_grbs];
  int mask[N_grbs,N_chan];
  vector[N_grbs] exposure;
}


transformed data{
  int all_N[N_grbs];
  int grainsize = 1;


  // precalculation of energy bounds

  for (n in 1:N_grbs) {

    all_N[n] = n;

  }
}

parameters{

  vector[N_grbs] index_raw;
  vector[N_grbs] log_K_raw; // raw energy flux norm

  real log_nH_host_mu_raw;
  
  real<lower=0> log_nH_host_sigma;
  vector[N_grbs] log_nH_host_raw;
  real<lower=0> index_sigma;
  real index_mu;


  
  // vector<lower=0>[N_grbs] nH_mw;

}


transformed parameters{
  vector[N_grbs] index;
  vector[N_grbs] log_K; // log eflux
  vector[N_grbs] K;
  vector[N_grbs] log_nH_host;
  vector[N_grbs] nH_host;
  vector[N_grbs] nH_host_norm;
  real log_nH_host_mu = log_nH_host_mu_raw + 22;
  
  
  log_nH_host = log_nH_host_mu + log_nH_host_raw * log_nH_host_sigma;
  index = index_mu + index_raw * index_sigma;

  log_K = log_K_raw -10;
  

  K =  pow(10, log_K);
  nH_host = pow(10, log_nH_host);
  nH_host_norm = nH_host  * inv(1e22);

}


model{


  index_raw ~ std_normal();
  log_K_raw ~ normal(0, 3);
  log_nH_host_raw ~ std_normal();

  log_nH_host_mu_raw ~ normal(0, 1);
  log_nH_host_sigma ~ normal(0, 1);

  index_mu ~ normal(-2, 1);
  index_sigma ~ std_normal();

  

  target += reduce_sum(partial_log_like, all_N, grainsize, N_ene, N_chan, host_precomputed_absorp, precomputed_absorp, ene_avg, ene_width, mask, n_chans_used, K, index, nH_host_norm, nH_mw, rsp, exposure, exposure_ratio, counts, bkg );

}
