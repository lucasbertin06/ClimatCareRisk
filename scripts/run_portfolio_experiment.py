from climacare.pipeline import open_pipeline
from climacare.inverse import run_map
from climacare.uq import laplace_posterior
from climacare.scenarios import posterior_theta_samples, simulate_scenarios
from src.loss_structure import optimize_portfolio, evaluate_smart_criterion, run_stress_tests, generate_efficient_frontier

with open_pipeline(config) as pipeline:
    map_result = run_map(pipeline, observations)
    laplace = laplace_posterior(pipeline, observations, map_result.estimate)
    thetas = posterior_theta_samples(laplace, rng_key, num_samples=200)
    batch = simulate_scenarios(pipeline, thetas, zones, cell_area)

    u_opt = optimize_portfolio(batch.scenarios_H_r, batch.scenarios_fire)
    smart = evaluate_smart_criterion(u_opt, batch.scenarios_H_r, batch.scenarios_fire, basis_noises)
    stress = run_stress_tests(u_opt, batch.scenarios_H_r, batch.scenarios_fire, basis_noises)
    frontier = generate_efficient_frontier(batch.scenarios_H_r, batch.scenarios_fire, n_points=20)