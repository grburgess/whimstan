real pll_no_whim(int [] n_slice,
                 int start,
                 int end,
                 int N_ene,
                 int N_chan,
                 vector[] host_precomputed_absorp,
                 vector[] precomputed_absorp,
                 vector[] ene_avg,
                 vector[] ene_width,
                 int[,] mask,
                 int[] n_chans_used,
                 vector K,
                 vector index,
                 vector nH_host,
                 vector[] mw_abs,
                 matrix[] rsp ,
                 vector exposure,
                 vector exposure_ratio,
                 vector[] counts,
                 vector[] bkg,
                 vector[] log_fact_obs,
                 vector[] log_fact_bkg,
                 vector[] o_plus_b,
                 vector[] alpha_bkg_factor


                 ){

  // host and mw ONLY absorption, fixed MW

  int slice_length = num_elements(n_slice);

  vector[slice_length] loglike;

  for (i in 1:slice_length){



    loglike[i] = cstat_optimized(counts[n,mask[n_slice[i],:n_chans_used[n_slice[i]]]],
                                 bkg[n_slice[i],mask[n_slice[i],:n_chans_used[n_slice[i]]]],
                                 ((rsp[n_slice[i]] * ((powerlaw_flux(ene_avg[n_slice[i]], K[n_slice[i]], index[n_slice[i]], 0.4, 15) .* absorption(nH_host[n_slice[i]], host_precomputed_absorp[n_slice[i]]) .* mw_abs[n_slice[i]]) .* ene_width[n_slice[i]])  * exposure[n_slice[i]]))[mask[n_slice[i],:n_chans_used[n_slice[i]]]],
                                 exposure_ratio[n_slice[i]],
                                 o_plus_b[n_slice[i],mask[n_slice[i],:n_chans_used[n_slice[i]]]],
                                 alpha_bkg_factor[n_slice[i],mask[n_slice[i],:n_chans_used[n_slice[i]]]],
                                 log_fact_obs[n_slice[i],mask[n_slice[i],:n_chans_used[n_slice[i]]]],
                                 log_fact_bkg[n_slice[i],mask[n_slice[i],:n_chans_used[n_slice[i]]]]
                                 );


  }

  return sum(loglike);

}



real pll_whim(int [] n_slice,
              int start,
              int end,
              int N_ene,
              int N_chan,
              vector[] ene_avg,
              vector[] ene_width,
              int[,] mask,
              int[] n_chans_used,
              vector[] mw_abs,
              vector K,
              vector index,
              real n0,
              matrix num,
              matrix[,] sum_sigma_interp,
              vector nH_host,
              vector[] host_precomputed_absorp,
              matrix[] rsp ,
              vector exposure,
              vector exposure_ratio,
              vector[] counts,
              vector[] bkg,
              vector[] log_fact_obs,
              vector[] log_fact_bkg,
              vector[] o_plus_b,
              vector[] alpha_bkg_factor
              ){

  int slice_length = num_elements(n_slice);

  vector[slice_length] loglike;

  for (i in 1:slice_length){

    // fill the log likelihood array


    loglike[i] =   cstat_optimized(counts[n,mask[n_slice[i],:n_chans_used[n_slice[i]]]],
                                   bkg[n_slice[i],mask[n_slice[i],:n_chans_used[n_slice[i]]]],
                                   (rsp[n_slice[i]] * ((powerlaw_flux(ene_avg[n_slice[i]], K[n_slice[i]], index[n_slice[i]], 0.4, 15) .* exp(integrate_absori_precalc(sum_sigma_interp[n_slice[i]], num, N_ene)*n0) .* mw_abs[n_slice[i]] .* absorption(nH_host[n_slice[i]], host_precomputed_absorp[n_slice[i]])) .* ene_width[n_slice[i]])  * exposure[n_slice[i]])[mask[n_slice[i],:n_chans_used[n_slice[i]]]],
                                   exposure_ratio[n_slice[i]],
                                   o_plus_b[n_slice[i],mask[n_slice[i],:n_chans_used[n_slice[i]]]],
                                   alpha_bkg_factor[n_slice[i],mask[n_slice[i],:n_chans_used[n_slice[i]]]],
                                   log_fact_obs[n_slice[i],mask[n_slice[i],:n_chans_used[n_slice[i]]]],
                                   log_fact_bkg[n_slice[i],mask[n_slice[i],:n_chans_used[n_slice[i]]]]
                                   );

  }

  return sum(loglike);

}
