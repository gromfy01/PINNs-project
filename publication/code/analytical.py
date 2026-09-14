# -*- coding: utf-8 -*-
"""
Аналитические модели напряжения волочения. Все формулы реализованы явно,
с указанием варианта записи. Единицы: alpha в РАДИАНАХ внутри функций.

r  — обжатие ПО ПЛОЩАДИ  r = 1 - (Rf/R0)^2
eps_hom = ln(A0/Af) = ln(1/(1-r)) = 2 ln(R0/Rf)
"""
import numpy as np

# ───────────────────────── SIEBEL ─────────────────────────
# Вариант S-A ("аддитивный", наиболее распространённый в англоязычной
# литературе; в задании дана именно эта форма):
#   sigma_d = sigma_f * [ ln(1/(1-r)) * (1 + mu/tan(alpha)) + (2/3)*alpha ]
# Вариант S-B (исходная запись Зибеля, малоугловое приближение cot a ~ 1/a):
#   sigma_d = sigma_f * [ ln(1/(1-r)) * (1 + mu/alpha)      + (2/3)*alpha ]
# Вариант S-C (некоторые учебники: редундант-член (2/3)*tan(alpha)):
#   sigma_d = sigma_f * [ ln(1/(1-r)) * (1 + mu/tan(alpha)) + (2/3)*tan(alpha) ]

def siebel(r, alpha, mu, sigma_f=1.0, variant="A"):
    eps = np.log(1.0 / (1.0 - r))
    if variant == "A":
        fr = 1.0 + mu / np.tan(alpha); red = (2.0 / 3.0) * alpha
    elif variant == "B":
        fr = 1.0 + mu / alpha;         red = (2.0 / 3.0) * alpha
    elif variant == "C":
        fr = 1.0 + mu / np.tan(alpha); red = (2.0 / 3.0) * np.tan(alpha)
    else:
        raise ValueError(variant)
    return sigma_f * (eps * fr + red)

# ───────────────────────── AVITZUR ─────────────────────────
# Верхнеграничное решение, сферическое поле скоростей, конический канал.
# Выведено заново (см. deriv_check ниже), совпадает с каноническим видом
# Avitzur (1963/1968):
#
#  sigma_xf/sigma_0 = 2 f(a) ln(R0/Rf)                       (пластическая работа)
#                   + (2/sqrt3) * (a - sin a cos a)/sin^2 a  (2 поверхности разрыва)
#                   + (2/sqrt3) * m * cot(a) * ln(R0/Rf)     (трение на конусе)
#                   + (2/sqrt3) * m * (L/Rf)                 (трение на пояске)
#
#  f(a) = [ 1 - cos a * sqrt(1 - (11/12) sin^2 a)
#           + (1/sqrt(132)) * ln( (1+sqrt(12/11)) / (cos a + sqrt(cos^2 a + 1/11)) ) ] / sin^2 a
#  f(0) = 1  (проверено численно)

S3 = np.sqrt(3.0)

def f_avitzur(alpha):
    a = np.asarray(alpha, dtype=float)
    s2 = np.sin(a) ** 2
    num = (1.0 - np.cos(a) * np.sqrt(1.0 - (11.0 / 12.0) * s2)
           + (1.0 / np.sqrt(132.0)) * np.log(
               (1.0 + np.sqrt(12.0 / 11.0)) /
               (np.cos(a) + np.sqrt(np.cos(a) ** 2 + 1.0 / 11.0))))
    out = np.where(s2 > 1e-12, num / np.where(s2 > 1e-12, s2, 1.0), 1.0)
    return out

def f_avitzur_numeric(alpha, n=200001):
    """I(a)/sin^2 a численно — независимая проверка замкнутой формы."""
    a = float(alpha)
    th = np.linspace(0.0, a, n)
    integ = 2.0 * np.sqrt(1.0 - (11.0 / 12.0) * np.sin(th) ** 2) * np.sin(th)
    return np.trapezoid(integ, th) / np.sin(a) ** 2

def avitzur(r, alpha, m, sigma_0=1.0, L_over_Rf=0.0):
    """m — фактор трения по касательному напряжению (tau = m*sigma0/sqrt3), 0..1."""
    lnR = 0.5 * np.log(1.0 / (1.0 - r))              # = ln(R0/Rf)
    a = alpha
    w_def = 2.0 * f_avitzur(a) * lnR
    w_shear = (2.0 / S3) * (a - np.sin(a) * np.cos(a)) / np.sin(a) ** 2
    w_fric = (2.0 / S3) * m * (np.cos(a) / np.sin(a)) * lnR
    w_land = (2.0 / S3) * m * L_over_Rf
    return sigma_0 * (w_def + w_shear + w_fric + w_land), dict(
        ideal=sigma_0 * 2.0 * lnR, redundant_def=sigma_0 * (w_def - 2.0 * lnR),
        shear=sigma_0 * w_shear, friction=sigma_0 * (w_fric + w_land))

# ─────────────── параметр неоднородности Δ (Backofen) ───────────────
# Δ = h_mean / L_contact = sin(a)*(R0+Rf)/(R0-Rf) = a*(1+sqrt(1-r))^2 / r  (малые a)
def delta_param(r, alpha, exact=True):
    Rf_R0 = np.sqrt(1.0 - r)
    if exact:
        return np.sin(alpha) * (1.0 + Rf_R0) / (1.0 - Rf_R0)
    return alpha * (1.0 + Rf_R0) ** 2 / r

if __name__ == "__main__":
    print("проверка f(alpha): замкнутая форма vs численное интегрирование")
    for adeg in [0.5, 1, 2, 4, 8, 12, 16, 20, 30, 45, 60]:
        a = np.radians(adeg)
        print(f"  a={adeg:5.1f} deg  f_closed={float(f_avitzur(a)):.8f}  f_num={f_avitzur_numeric(a):.8f}")
    print("  f(a->0) closed =", float(f_avitzur(1e-6)))
    print()
    print("предельная проверка Авитцура при a->0, m=0: должно быть ln(1/(1-r))")
    for r in [0.03, 0.20, 0.45]:
        v,_ = avitzur(r, np.radians(0.01), 0.0)
        print(f"  r={r:.2f}  avitzur={v:.6f}  ln(1/(1-r))={np.log(1/(1-r)):.6f}")
    print()
    print("Δ: точная и малоугловая формы")
    for r,adeg in [(0.03,20),(0.20,12),(0.45,4)]:
        a=np.radians(adeg)
        print(f"  r={r:.2f} a={adeg}deg  Δ_exact={delta_param(r,a,True):.4f}  Δ_small={delta_param(r,a,False):.4f}")
