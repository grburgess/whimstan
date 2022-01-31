vector logfactorial(vector x){

  return lgamma(x +1);

}




real cstat_optimized_vec(vector observed_counts,
                         vector background_counts,
                         vector predicted_counts,
                         real alpha,
                         vector o_plus_b,
                         vector alpha_bkg_factor,
                         vector log_fact_obs,
                         vector log_fact_bkg,
			 vector pad) {


  int N = num_elements(predicted_counts);

  vector[N] B_mle =  inv(2.0 * alpha * (1 + alpha))
    * (alpha * (o_plus_b) - (alpha + 1) * predicted_counts +  sqrt(
                                                                   alpha_bkg_factor .* predicted_counts
                                                                   + square((alpha + 1) * predicted_counts - alpha * (o_plus_b))
                                                                   )   );


  return sum(lmultiply( observed_counts, alpha * B_mle + predicted_counts ) + lmultiply(background_counts, B_mle + pad)
	     - (alpha + 1) * B_mle - predicted_counts - log_fact_bkg - log_fact_obs);


}




real cstat_optimized(vector observed_counts,
                     vector background_counts,
                     vector predicted_counts,
                     real alpha,
                     vector o_plus_b,
                     vector alpha_bkg_factor,
                     vector log_fact_obs,
                     vector log_fact_bkg) {

  int N = num_elements(predicted_counts);
  vector[N] loglike;

  for (n in 1:N) {



    // real sqr = sqrt(
    //                 alpha_bkg_factor[n] * predicted_counts[n]
    //                 + square((alpha + 1) * predicted_counts[n] - alpha * (o_plus_b[n]))
    //                 );

    // profiled background

    real B_mle =  inv(2.0 * alpha * (1 + alpha))
      * (alpha * (o_plus_b[n]) - (alpha + 1) * predicted_counts[n] +  sqrt(
                                                                           alpha_bkg_factor[n] * predicted_counts[n]
                                                                           + square((alpha + 1) * predicted_counts[n] - alpha * (o_plus_b[n]))
                                                                           )   );



    if(background_counts[n] > 0){

      loglike[n] = lmultiply( observed_counts[n], alpha *  B_mle+ predicted_counts[n] ) + lmultiply(background_counts[n], B_mle)
        - (alpha + 1) * B_mle - predicted_counts[n] - log_fact_bkg[n] - log_fact_obs[n];

    }

    else {

      loglike[n] = lmultiply( observed_counts[n], alpha *  B_mle+ predicted_counts[n] ) +
        - (alpha + 1) * B_mle - predicted_counts[n] -  log_fact_obs[n];




    }



  }



  return sum(loglike);





}
