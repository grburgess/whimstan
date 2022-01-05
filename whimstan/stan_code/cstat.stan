real logfactorial(real x){

  return lgamma(x +1);

}





real cstat(vector observed_counts, vector background_counts, vector predicted_counts, real alpha) {

  int N = num_elements(predicted_counts);
  real loglike =0;

  vector[N] o_plus_b = observed_counts + background_counts;

  for (n in 1:N) {



    real sqr = sqrt(
                    4 * (alpha + square(alpha)) * background_counts[n] * predicted_counts[n]
                    + square((alpha + 1) * predicted_counts[n] - alpha * (o_plus_b[n]))
                    );

    real B_mle =  inv(2.0 * alpha * (1 + alpha))
      * (alpha * (o_plus_b[n]) - (alpha + 1) * predicted_counts[n] + sqr);
    //   // Profile likelihood


    if(background_counts[n] > 0){

      loglike += lmultiply( observed_counts[n], alpha *  B_mle+ predicted_counts[n] ) + lmultiply(background_counts[n], B_mle)
        - (alpha + 1) * B_mle - predicted_counts[n] - logfactorial(background_counts[n]) - logfactorial(observed_counts[n]);

    }

    else {


      loglike += lmultiply( observed_counts[n], alpha *  B_mle+ predicted_counts[n] ) +
        - (alpha + 1) * B_mle - predicted_counts[n] -  logfactorial(observed_counts[n]);




    }



  }



  return loglike;





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


  //  vector[N] sqr =

  for (n in 1:N) {



    real sqr = sqrt(
                    alpha_bkg_factor[n] * predicted_counts[n]
                    + square((alpha + 1) * predicted_counts[n] - alpha * (o_plus_b[n]))
                    );

    // profiled background

    real B_mle =  inv(2.0 * alpha * (1 + alpha))
      * (alpha * (o_plus_b[n]) - (alpha + 1) * predicted_counts[n] + sqr);



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
