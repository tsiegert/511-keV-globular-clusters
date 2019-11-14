data {

  int<lower=0> N;             // number of data points
  vector<lower=0>[N] x_obs;            // x observations
  vector<lower=0>[N] y_obs;            // y observations  
  vector<lower=0>[N] sigma_x; // measurement uncertainty in x
  vector<lower=0>[N] sigma_y; // measurement uncertainty in y
  vector<lower=0>[N] dist_obs;         // distance to objects

  
  int<lower=0> N_model;       // number of point to evaluate the fitted model
  vector<lower=0>[N_model] x_model;    // where to evaluate the model

  //priors on the fitted parameters
  real mu_lgNorm;
  real sigma_lgNorm;
  real mu_alpha;
  real sigma_alpha;

}

parameters {
  
  real lgNorm;  // normalisation of power-law function
  real alpha;   // power-law index
  vector<lower=0>[N] lgx_latent; // latent x location (x_obs + randomn(sigma_x))
  real<lower=1> sigma;   // scale for "systematic uncertainty"
  
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
      lgsigma_y[n] = (fabs(log10(y_obs[n]) - log10(fabs(y_obs[n]-sigma_y[n]))) + fabs(log10(fabs(y_obs[n]+sigma_y[n])) - log10(y_obs[n])))/2.;
      lgsigma_x[n] = (fabs(log10(x_obs[n]) - log10(fabs(x_obs[n]-sigma_x[n]))) + fabs(log10(fabs(x_obs[n]+sigma_x[n])) - log10(x_obs[n])))/2.;
  }
}

model {

  // priors
  lgNorm ~ normal(mu_lgNorm,sigma_lgNorm);
  alpha ~ normal(mu_alpha,sigma_alpha);
  sigma ~ cauchy(1.,5.);
  
  lgx_latent ~ normal(lgx_obs,lgsigma_x*sigma);

  // likelihood

  lgy_obs ~ normal(lgy_true, lgsigma_y*sigma);
  
 
}

generated quantities {

  vector[N] ppc_x;
  vector[N] ppc_y;
  vector[N] five11_flux;
  
  vector[N_model] powerlaw;
    
  real<lower=0> GC511_flux_total = 0.;

  // generate posteriors
  // fitted powerlaw
  for (nm in 1:N_model) {
      powerlaw[nm] = pow(10.,lgNorm) * pow(x_model[nm],alpha);
  }

  // create posterior samples for PPC
  for (n in 1:N) {

    ppc_x[n] = pow(10.,normal_rng(lgx_latent[n], lgsigma_x[n]*sigma));
    ppc_y[n] = pow(10.,normal_rng(lgNorm + lgx_latent[n]*alpha, lgsigma_y[n]*sigma));

    // prediction of 511 keV flux assuming conversion factor bewteen GeV and 511 keV as (1.5+-0.1)e-3 GeV/ph (Bartels+2018)
    // or in luminosity in the bulge: 2e37 erg/s (+-10%) a 0.96e-3 ph/cm2/s (+-5%) == 6.3e36 erg/s --> (0.32=-0.03) erg/s / (erg/s)
    // 4 pi kpc**2 == 1.1965e44 cm**2
    // 1 erg == 1221430 (511 keV) photons
    five11_flux[n] = ppc_y[n]*normal_rng(0.32,0.03)/(dist_obs[n]^2*1.1965e44)*1.22143e6;
    GC511_flux_total += five11_flux[n];
    
  }
  

}