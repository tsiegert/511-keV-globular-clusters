data {

  int<lower=0> N;
  vector<lower=0>[N] sigma_y_percent;
  vector<lower=0>[N] sigma_x_percent;
  real lgNorm;
  real alpha;
  real lgx_min;
  real lgx_max;

}


generated quantities {
  
  // storage for the generated data
  vector<lower=0>[N] x_obs;
  vector<lower=0>[N] y_obs;
  vector[N] x_latent;
  vector[N] y_latent;
  
  for (n in 1:N) {

    // randomly pull x from a uniform distribution
    x_latent[n] = pow(10.,uniform_rng(lgx_min, lgx_max));

    // calculate the latent y 
    y_latent[n] = pow(10.,lgNorm) * pow(x_latent[n],alpha);

    // obscure y with measurement error
    y_obs[n] = normal_rng(y_latent[n], sigma_y_percent[n]*y_latent[n]);
    // obscure y with measurement error
    x_obs[n] = normal_rng(x_latent[n], sigma_x_percent[n]*x_latent[n]);

        
  }
}