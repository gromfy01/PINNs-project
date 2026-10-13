# Cover letter — Engineering with Computers

<!-- Draft 13.10.2026. Placeholders in angle brackets: <URL> of a clean public archive of the code and data
(not the working repository, which holds internal notes), and the competing-interests statement. -->

Dear Professor Zhang,

We submit the manuscript **"Physics-informed versus data-driven surrogates for residual stress after wire drawing: accuracy, physical admissibility and behaviour outside the training range"** for consideration as an original research article in *Engineering with Computers*.

The manuscript addresses a question within the journal's scope of *data-driven and hybrid modeling methods*: what a physics term in the loss adds to a neural-network surrogate of finite-element results. A multilayer perceptron (MLP), a strong-form physics-informed network (PINN) and a weak-form variational PINN (VPINN) share one architecture, one optimiser and one training budget and differ only in the loss. The contribution is the evaluation protocol and what it shows, not the application of an established method to a new process. The protocol pairs each extrapolation split with a random control of the same size, compares the families over seeds (five in the one-dimensional track, three in the extended one) with seed-paired tests and a Holm correction, and audits the equilibrium residuals and the traction-free condition of the predicted fields with the same discrete operators that it applies to the finite-element source. Apart from the governing equations, none of these elements is specific to wire drawing. The finite-element simulations, 1756 axisymmetric runs of single-pass drawing of AISI 1020 steel, serve as the data source, with the drawing stress checked against the relations of Siebel and Avitzur on 112 dedicated runs. The code and the raw finite-element reports are available at <URL>.

The main findings are as follows.

- **Accuracy.** The PINN has a lower error than the MLP on all thirteen splits of the one-dimensional track, by under 0.4 MPa on the random, interpolation and matched-control splits, by 0.6–1.9 MPa on three single-factor extrapolations and a gap in the joint semi-angle–reduction range, and by 7.5 MPa at the highest speed, where all errors are 48–56 MPa. The difference is significant on four splits and on none after the Holm correction. With the axial coordinate as input, the PINN is 0.2 MPa less accurate than the MLP on the random split, a difference that is not significant.
- **Admissibility.** On the random split and the die semi-angle extrapolation the PINN lowers the radial equilibrium residual of the MLP by 40–52 % in both tracks. In the extended track both forms of the PINN predict a radial stress at the free surface 3.5–6.3 times smaller than that of the finite-element source on the same held-out cases, whose surface values carry the error of nodal extrapolation. The axial residual falls significantly, by 33–41 %, only when the axial equation is in the loss.
- **Label defects.** With a +250 MPa offset of the hoop stress in 10 % of the training cases, the only defect tested, the error grows 1.64-fold for the PINN against 1.81-fold for the MLP. After the Holm correction the gap between them remains significant only at 5 % corruption.

The study follows two papers by the first author and co-workers in *The International Journal of Advanced Manufacturing Technology* (2024, 133:3461–3473, and 2025, 141:1847–1858), in which purely data-driven networks trained on finite-element results predicted the residual stresses after wire drawing. The present manuscript uses a new finite-element dataset, computed for a different material, takes the data-driven network as the baseline and examines what a physics term adds, together with the admissibility of the predicted fields and the behaviour outside the training range. The relation is stated in the Introduction.

The manuscript has not been published and is not under consideration elsewhere. Both authors have approved the submission and declare no competing interests <to be confirmed>.

Yours sincerely,

Dmitriy Demin (corresponding author)
HSE University, 11 Pokrovsky Boulevard, Moscow, Russia
ddemin@hse.ru

on behalf of the authors, Dmitriy Demin and Georgii Romanenko
