data {

  int<lower=0> N;             // number of data points
  vector[N] x_obs;            // x observations
  vector[N] y_obs;            // y observations  
  vector<lower=0>[N] sigma_x; // measurement uncertainty in x
  vector<lower=0>[N] sigma_y; // measurement uncertainty in y

  
  int<lower=0> N_model;       // number of point to evaluate the fitted model
  vector[N_model] x_model;    // where to evaluate the model

  //priors on the fitted parameters
  real mu_lgNorm;
  real sigma_lgNorm;
  real mu_alpha;
  real sigma_alpha;

}

parameters {
  
  real lgNorm;  // normalisation of power-law function
  real alpha; // power-law index
  vector[N] lgx_latent; // latent x location (x_obs + randomn(sigma_x))
  
}

transformed parameters {

  // observed lgy values
  vector[N] lgy_obs = log10(y_obs);
  // latent lgy values, not obscured by measurement error
  vector[N] lgy_true = lgNorm + lgx_latent*alpha;
  // y and x uncertainty in dex
  vector[N] lgsigma_y; 
  vector[N] lgsigma_x;
  // observed lgx values
  vector[N] lgx_obs = log10(x_obs);
  // estimate for y and x uncertainty in dex
  for (n in 1:N) {
      lgsigma_y[n] = (abs(log10(y_obs[n]) - log10(abs(y_obs[n]-sigma_y[n]))) + abs(log10(abs(y_obs[n]+sigma_y[n])) - log10(y_obs[n])))/2.;
      lgsigma_x[n] = (abs(log10(x_obs[n]) - log10(abs(x_obs[n]-sigma_x[n]))) + abs(log10(abs(x_obs[n]+sigma_x[n])) - log10(x_obs[n])))/2.;
  }
}

model {

  // priors
  lgNorm ~ normal(mu_lgNorm,sigma_lgNorm);
  alpha ~ normal(mu_alpha,sigma_alpha);
  lgx_latent ~ normal(lgx_obs,lgsigma_x);

  // likelihood

  lgy_obs ~ normal(lgy_true, lgsigma_y);
  
 
}

generated quantities {

  vector[N] ppc_x;
  vector[N] ppc_y;
  
  vector[N_model] powerlaw;

  // generate posteriors
  // fitted powerlaw
  for (nm in 1:N_model) {
      powerlaw[nm] = pow(10.,lgNorm) * pow(x_model[nm],alpha);
  }

  // create posterior samples for PPC
  for (n in 1:N) {

    ppc_x[n] = pow(10.,normal_rng(lgx_latent[n], lgsigma_x[n]));
    ppc_y[n] = pow(10.,normal_rng(lgNorm + lgx_latent[n]*alpha, lgsigma_y[n]));
    
  }
  

}