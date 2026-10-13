# Cover letter — Engineering with Computers

Dear Professor Zhang,

We submit the manuscript **"Physics-informed versus data-driven surrogates for residual stress after wire drawing: accuracy, physical admissibility and behaviour outside the training range"** for consideration as an original research article in *Engineering with Computers*.

Physics-informed neural networks are increasingly proposed as surrogates of finite-element simulations, but their benefit over a purely data-driven network is usually shown with different architectures, optimisers or training budgets, on a single random hold-out and by accuracy alone. Our manuscript is a controlled comparison designed to separate the effect of the physics term from everything else. A multilayer perceptron, a strong-form PINN and a weak-form variational PINN share one architecture, one optimiser and one training budget and differ only in the loss. They are trained on 1756 axisymmetric finite-element simulations of wire drawing and compared on thirteen interpolation and extrapolation splits with matched controls, over several random seeds, with paired tests and a Holm correction. Besides accuracy, we audit the physical admissibility of every predicted field, the equilibrium residuals and the violation of the traction-free condition, with the same operators applied to the finite-element source itself. A two-dimensional extension with the axial coordinate as input, a label-corruption test and a comparison of two automatic-differentiation back-ends complete the study.

The main findings are the following.

- The physics term does not buy accuracy. The PINN has a lower error than the MLP on all thirteen one-dimensional splits, but by under 0.4 MPa on the random, interpolation and matched-control splits, and none of the differences survives the Holm correction. In the two-dimensional extension the PINN is 0.2 MPa less accurate than the MLP on the random split, a difference that is not significant.
- The physics term buys admissibility. The PINN lowers the radial equilibrium residual of the MLP by 40–52 %, and the violation of the traction-free condition falls 3.9–5.3-fold in the one-dimensional track. In the two-dimensional extension both forms of the PINN satisfy the traction-free condition 3.5–6.3 times more strictly than the finite-element data they were trained on. The axial residual falls only when the axial equation is in the loss.
- Under a constant offset added to one stress component in 10 % of the training cases, the error of the PINN grows 1.64-fold against 1.81-fold for the MLP.

We believe the manuscript fits the scope of the journal under *data-driven and hybrid modeling methods*. Its contribution is not the application of an established method to a new process but a reusable protocol for evaluating physics-informed surrogates of simulation data: an identical training setup, extrapolation splits with matched controls, seed statistics with correction for multiple comparisons, and an admissibility audit measured against the simulation itself. The finite-element campaign is the data source. Its drawing stress is verified against the analytical relations of Siebel and Avitzur on 112 dedicated runs, and its cleaning is documented case by case. The code and the raw finite-element reports are publicly available [repository link], in line with the journal's encouragement to make codes available.

The study builds on two earlier papers of the authors in *The International Journal of Advanced Manufacturing Technology* (2024, 2025), which showed that data-driven networks reproduce the residual stresses after wire drawing with high accuracy. The present work does not repeat them. It fixes the network and examines the effect of the physics term, the admissibility of the predicted fields and the behaviour outside the training range, which the earlier papers did not address. The relation is stated explicitly in the Introduction.

The manuscript has not been published and is not under consideration elsewhere. All authors have approved the submission and declare no competing interests [to be confirmed].

[Optional: suggested reviewers, 3–5 names with affiliations and e-mail addresses.]

Yours sincerely,

Dmitriy Demin (corresponding author)
HSE University, 11 Pokrovsky Boulevard, Moscow, Russia
ddemin@hse.ru

on behalf of the authors, Dmitriy Demin and Georgii Romanenko
