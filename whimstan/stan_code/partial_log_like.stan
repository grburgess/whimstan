real partial_log_like(int [] n_slice, int start, int end, int N_ene, int N_chan, vector[] host_precomputed_absorp, vector[] precomputed_absorp, vector[] ene_avg, vector[] ene_width, int[,] mask, int[] n_chans_used, vector K, vector index, vector nH_host, vector nH_mw, matrix[] rsp , vector exposure, vector exposure_ratio, vector[] counts, vector[] bkg){
  int slice_length = num_elements(n_slice);

  real loglike = 0;

  for (i in 1:slice_length){

    int n = n_slice[i];


    vector[N_ene] source_spectrum = powerlaw_flux(ene_avg[n], K[n], index[n], 0.4, 15) .* absorption(nH_host[n], host_precomputed_absorp[n]) .* absorption(nH_mw[n], precomputed_absorp[n]);


    vector[N_ene] integral_flux = source_spectrum .* ene_width[n];

    vector[N_chan] predicted_counts =  (rsp[n] * integral_flux)  * exposure[n];
    // print(predicted_counts);
    loglike +=  cstat(counts[n,mask[n,:n_chans_used[n]]],
                      bkg[n,mask[n,:n_chans_used[n]]],
                      predicted_counts[mask[n,:n_chans_used[n]]],
                      exposure_ratio[n]);

  }
  return loglike;

}

real partial_log_like_whimonly(int [] n_slice, int start, int end, int N_ene, int N_chan, vector[] ene_avg, vector[] ene_width, int[,] mask, int[] n_chans_used, vector K, vector index, real n0, matrix num, matrix[,] sum_sigma_interp,  matrix[] rsp , vector exposure, vector exposure_ratio, vector[] counts, vector[] bkg){
  int slice_length = num_elements(n_slice);

  real loglike = 0;

  for (i in 1:slice_length){

    int n = n_slice[i];


    vector[N_ene] source_spectrum = powerlaw_flux(ene_avg[n], K[n], index[n], 0.4, 15) .* integrate_absori(sum_sigma_interp[n], num, n0, N_ene);


    vector[N_ene] integral_flux = source_spectrum .* ene_width[n];
    vector[N_chan] predicted_counts =  (rsp[n] * integral_flux)  * exposure[n];
    // print(predicted_counts);
    loglike +=  cstat(counts[n,mask[n,:n_chans_used[n]]],
                      bkg[n,mask[n,:n_chans_used[n]]],
                      predicted_counts[mask[n,:n_chans_used[n]]],
                      exposure_ratio[n]);

  }

  return loglike;

}

real partial_log_like_whimonly_t_fixed(int [] n_slice, int start, int end, int N_ene, int N_chan, vector[] ene_avg, vector[] ene_width, int[,] mask, int[] n_chans_used, vector K, vector index, real n0, vector[] whim_abs_base,  matrix[] rsp , vector exposure, vector exposure_ratio, vector[] counts, vector[] bkg){
  int slice_length = num_elements(n_slice);

  real loglike = 0;

  for (i in 1:slice_length){

    int n = n_slice[i];


    vector[N_ene] source_spectrum = powerlaw_flux(ene_avg[n], K[n], index[n], 0.4, 15) .* exp(whim_abs_base[n]*n0);


    vector[N_ene] integral_flux = source_spectrum .* ene_width[n];
    vector[N_chan] predicted_counts =  (rsp[n] * integral_flux)  * exposure[n];
    // print(predicted_counts);
    loglike +=  cstat(counts[n,mask[n,:n_chans_used[n]]],
                      bkg[n,mask[n,:n_chans_used[n]]],
                      predicted_counts[mask[n,:n_chans_used[n]]],
                      exposure_ratio[n]);

  }

  return loglike;

}

real partial_log_like_whim_and_mw_t_fixed(int [] n_slice, int start, int end, int N_ene, int N_chan, vector[] ene_avg, vector[] ene_width, int[,] mask, int[] n_chans_used, vector[] mw_abs, vector K, vector index, real n0, vector[] whim_abs_base,  matrix[] rsp , vector exposure, vector exposure_ratio, vector[] counts, vector[] bkg){
  int slice_length = num_elements(n_slice);

  real loglike = 0;

  for (i in 1:slice_length){

    int n = n_slice[i];


    vector[N_ene] source_spectrum = powerlaw_flux(ene_avg[n], K[n], index[n], 0.4, 15) .* exp(whim_abs_base[n]*n0) .* mw_abs[n];


    vector[N_ene] integral_flux = source_spectrum .* ene_width[n];
    vector[N_chan] predicted_counts =  (rsp[n] * integral_flux)  * exposure[n];
    // print(predicted_counts);
    loglike +=  cstat(counts[n,mask[n,:n_chans_used[n]]],
                      bkg[n,mask[n,:n_chans_used[n]]],
                      predicted_counts[mask[n,:n_chans_used[n]]],
                      exposure_ratio[n]);

  }

  return loglike;

}

real partial_log_like_all_t_fixed(int [] n_slice, int start, int end, int N_ene, int N_chan, vector[] ene_avg, vector[] ene_width, int[,] mask, int[] n_chans_used, vector[] mw_abs, vector K, vector index, real n0, vector[] whim_abs_base, vector nH_host, vector[] host_precomputed_absorp, matrix[] rsp , vector exposure, vector exposure_ratio, vector[] counts, vector[] bkg){
  int slice_length = num_elements(n_slice);

  real loglike = 0;

  for (i in 1:slice_length){

    int n = n_slice[i];


    vector[N_ene] source_spectrum = powerlaw_flux(ene_avg[n], K[n], index[n], 0.4, 15) .* exp(whim_abs_base[n]*n0) .* mw_abs[n] .* absorption(nH_host[n], host_precomputed_absorp[n]);


    vector[N_ene] integral_flux = source_spectrum .* ene_width[n];
    vector[N_chan] predicted_counts =  (rsp[n] * integral_flux)  * exposure[n];
    // print(predicted_counts);
    loglike +=  cstat(counts[n,mask[n,:n_chans_used[n]]],
                      bkg[n,mask[n,:n_chans_used[n]]],
                      predicted_counts[mask[n,:n_chans_used[n]]],
                      exposure_ratio[n]);

  }

  return loglike;

}

real partial_log_like_all(int [] n_slice, int start, int end, int N_ene, int N_chan, vector[] ene_avg, vector[] ene_width, int[,] mask, int[] n_chans_used, vector[] mw_abs, vector K, vector index, real n0, matrix num, matrix[,] sum_sigma_interp, vector nH_host, vector[] host_precomputed_absorp, matrix[] rsp , vector exposure, vector exposure_ratio, vector[] counts, vector[] bkg){
  int slice_length = num_elements(n_slice);

  real loglike = 0;

  for (i in 1:slice_length){

    int n = n_slice[i];


    vector[N_ene] source_spectrum = powerlaw_flux(ene_avg[n], K[n], index[n], 0.4, 15) .* exp(integrate_absori_precalc(sum_sigma_interp[n], num, N_ene)*n0) .* mw_abs[n] .* absorption(nH_host[n], host_precomputed_absorp[n]);


    vector[N_ene] integral_flux = source_spectrum .* ene_width[n];
    vector[N_chan] predicted_counts =  (rsp[n] * integral_flux)  * exposure[n];
    // print(predicted_counts);
    loglike +=  cstat(counts[n,mask[n,:n_chans_used[n]]],
                      bkg[n,mask[n,:n_chans_used[n]]],
                      predicted_counts[mask[n,:n_chans_used[n]]],
                      exposure_ratio[n]);

  }

  return loglike;

}
