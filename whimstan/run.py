from cmdstanpy import CmdStanModel

from prep import build_stan_data

data = build_stan_data("050401")

model = CmdStanModel(model_name="xrt", stan_file="simple_xrt.stan")
model.compile(force=True)

fit = model.sample(data=data,
                   chains=1,
                   seed=1234,
                   iter_warmup=200,
                   iter_sampling=200,
                   show_progress=True,





                   )
