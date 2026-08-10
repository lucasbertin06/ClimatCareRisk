# Spécification mathématique — tranche C0

**Projet :** ClimaCare-Risk  
**Version :** proposition 0.1  
**Statut :** en attente de validation ; aucune implémentation scientifique ne
doit précéder son approbation

## 1. Objet et limites

Cette spécification définit un modèle mécaniste, adimensionné, différentiable
et reproductible sur le domaine

\[
\Omega=[0,1]\times[0,1],\qquad t\in[0,T_f].
\]

Il s'agit d'un banc d'essai numérique pour l'inférence différentiable, et non
d'un modèle quantitatif d'incendie, d'une prévision clinique ou d'un modèle
actuariel validé. Les coefficients santé et finance sont synthétiques.

Le chemin inverse C0 est :

\[
\theta
\longmapsto \mathrm{FireSpread}(\theta)
\longmapsto S
\longmapsto \mathrm{SmokeTransport}(S,\theta)
\longmapsto \widehat y
\longmapsto \mathcal L_{\mathrm{MAP}}.
\]

La santé et la finance sont des fonctions JAX natives en aval. Elles fournissent
des diagnostics et des tests C0, mais ne biaisent pas la vraisemblance du premier
problème inverse.

## 2. Variables et paramètres

### 2.1 Champs

| Symbole | Domaine | Définition |
|---|---|---|
| \(T(x,t)\) | \([0,+\infty)\) | intensité thermique normalisée |
| \(F(x,t)\) | \([0,1]\) | fraction de combustible restante |
| \(M(x)\) | \([0,1]\) | humidité normalisée, fixée dans C0 |
| \(S(x,t)\) | \([0,+\infty)\) | source normalisée de fumée |
| \(c(x,t)\) | \([0,+\infty)\) | concentration normalisée de fumée |
| \(p_r(x)\) | \([0,+\infty)\) | densité de population synthétique de la zone \(r\) |
| \(u_{\mathrm{fuel}}(x)\) | \([0,1]\) | niveau continu de prévention combustible |

### 2.2 Paramètres inférés

Le premier vecteur physique est strictement limité à

\[
\theta=(x_0,y_0,\log A_0,\delta_\phi).
\]

- \(x_0,y_0\) : position continue de l'ignition ;
- \(A_0=\exp(\log A_0)>0\) : amplitude initiale ;
- \(\delta_\phi\) : correction bornée de la direction du vent.

Pour le gradient check, \(\theta\) est fourni en coordonnées physiques et choisi
strictement à l'intérieur de ses bornes.

Pour l'optimisation MAP, des variables libres \(z\in\mathbb R^4\) sont décodées
par

\[
\begin{aligned}
x_0 &= m_x+(1-2m_x)\,\sigma(z_x),\\
y_0 &= m_y+(1-2m_y)\,\sigma(z_y),\\
\log A_0 &= \mu_A+s_Az_A,\\
\delta_\phi &= \delta_{\max}\tanh(z_\phi),
\end{aligned}
\]

où \(m_x,m_y\) sont des marges de grille et \(\sigma\) la sigmoïde logistique. Le
rapport distinguera toujours gradients physiques \(\nabla_\theta\mathcal L\)
et gradients d'optimisation \(\nabla_z\mathcal L\).

Le décodage est ici un outil de contrainte numérique : la loss optimisée reste
le log-posterior défini dans les coordonnées physiques \(\theta\). Aucun terme
de Jacobien de changement de variable n'est ajouté au MAP physique.

### 2.3 Paramètres fixés dans le premier cas

\[
D_T,\ D_c,\ \eta_{\mathrm{smoke}},\ \lambda_c,\ M,\ h,\ k_r,\ Q,\
\alpha_M,\ T_{\mathrm{ign}},\ \varepsilon_T,\ \sigma_0,
\]

ainsi que la norme de chaque vent, les biais capteurs et les écarts-types de
bruit. Les biais valent zéro dans Tiny et les écarts-types sont connus.

Il est interdit d'inférer simultanément \(A_0\) et
\(\eta_{\mathrm{smoke}}\) dans C0.

## 3. Vent partagé

Soit \(\phi_b\) l'angle de base et

\[
d(\delta_\phi)=
\begin{bmatrix}
\cos(\phi_b+\delta_\phi)\\
\sin(\phi_b+\delta_\phi)
\end{bmatrix}.
\]

Les vitesses sont

\[
v=s_Td(\delta_\phi),\qquad
w=s_cd(\delta_\phi),
\]

avec normes \(s_T,s_c\) fixées. La correction \(\delta_\phi\) agit donc sur les
deux Tesseracts. Son gradient total additionne le chemin par la propagation du
feu et le chemin direct par le transport de fumée.

Dans Tiny, les bornes de \(\delta_\phi\) sont choisies de sorte que les deux
composantes du vent restent éloignées de zéro. Cela évite le point non
différentiable du découpage upwind \(w^+=\max(w,0)\),
\(w^-=\min(w,0)\).

## 4. Tesseract A — propagation du feu

### 4.1 Modèle continu

\[
\frac{\partial T}{\partial t}
=
\nabla\cdot(D_T\nabla T)
-v\cdot\nabla T
-hT
+QR(T,F,M),
\]

\[
\frac{\partial F}{\partial t}=-R(T,F,M),
\]

avec

\[
R(T,F,M)=
k_rF\exp(-\alpha_MM)
\operatorname{sigmoid}
\left(\frac{T-T_{\mathrm{ign}}}{\varepsilon_T}\right).
\]

Contraintes :

\[
D_T>0,\quad h\ge 0,\quad k_r>0,\quad Q>0,\quad
\varepsilon_T>0.
\]

### 4.2 Conditions initiales

Le combustible initial est

\[
F(x,0)=F_{\mathrm{base}}(x)
\left[1-u_{\mathrm{fuel}}(x)\right],
\qquad 0\le F_{\mathrm{base}},u_{\mathrm{fuel}}\le1.
\]

Le cas Tiny utilise \(F_{\mathrm{base}}=1\) et
\(u_{\mathrm{fuel}}=0\).

L'intensité initiale différentiable est

\[
T(x,0)=A_0
\exp\left(
-\frac{\lVert x-(x_0,y_0)\rVert^2}{2\sigma_0^2}
\right).
\]

Aucun seuillage binaire n'est autorisé.

### 4.3 Source de fumée

À chaque pas direct,

\[
S(x,t)=\eta_{\mathrm{smoke}}R(T,F,M).
\]

Le tenseur complet \(S[n,j,i]\) est transmis au Tesseract B. Il n'est ni réduit
à un centre de masse, ni remplacé par son maximum temporel.

### 4.4 Conditions aux limites

- diffusion thermique : flux normal nul,
  \(\nabla T\cdot n=0\) ;
- advection : valeur extérieure homogène à l'entrée et sortie convective
  discrète à l'aval ;
- le combustible ne possède pas de flux spatial, car son équation est locale.

Ces conditions doivent être encodées explicitement, pas obtenues implicitement
par un padding non documenté.

### 4.5 Discrétisation

La grille est cartésienne, uniforme et centrée en cellules :

\[
x_i=(i+\tfrac12)\Delta x,\qquad
y_j=(j+\tfrac12)\Delta y,
\]

\[
\Delta x=1/N_x,\qquad \Delta y=1/N_y.
\]

Euler explicite est utilisé. Pour \(n=0,\ldots,N_t-1\),

\[
\begin{aligned}
R^n_{ij}
&=
k_rF^n_{ij}e^{-\alpha_MM_{ij}}
\sigma\left(
\frac{T^n_{ij}-T_{\mathrm{ign}}}{\varepsilon_T}
\right),\\
S^n_{ij}&=\eta_{\mathrm{smoke}}R^n_{ij},\\
T^{n+1}
&=
T^n+\Delta t
\left(
D_T\Delta_hT^n
-v\cdot\nabla_h^{\mathrm{up}}T^n
-hT^n+QR^n
\right),\\
F^{n+1}&=F^n-\Delta t\,R^n.
\end{aligned}
\]

\(\Delta_h\) est le Laplacien centré à cinq points et
\(\nabla_h^{\mathrm{up}}\) l'opérateur upwind du premier ordre.

### 4.6 Positivité, clamps et stabilité

Le schéma nominal C0 n'utilise aucun clamp sur \(T\) ou \(F\). La positivité est
obtenue par des contraintes suffisantes :

\[
\nu_T=
\Delta t\left[
\frac{|v_x|}{\Delta x}
+\frac{|v_y|}{\Delta y}
+2D_T\left(\frac1{\Delta x^2}+\frac1{\Delta y^2}\right)
+h
\right]\le 1,
\]

\[
\Delta t\,k_r\le1.
\]

En effet,

\[
F^{n+1}=F^n
\left[
1-\Delta t\,k_re^{-\alpha_MM}
\sigma\left(
\frac{T-T_{\mathrm{ign}}}{\varepsilon_T}
\right)
\right]\in[0,F^n].
\]

Le programme doit calculer \(\nu_T\) avant la simulation et lever une erreur
explicite si une condition est violée. Une tolérance machine peut être utilisée
dans les assertions, jamais pour masquer une instabilité.

Si un clamp de sécurité devenait nécessaire après tests, il devrait faire
l'objet d'une modification explicite de cette spécification et d'un test
montrant que le point du gradient check n'est pas saturé.

### 4.7 Sorties

- intensité \(T\) aux instants configurés ;
- combustible final \(F^{N_t}\) ;
- source complète \(S[0:N_t,:,:]\) ;
- fraction brûlée finale

\[
A_{\mathrm{burned}}
=\Delta x\Delta y\sum_{ij}
\left(F^0_{ij}-F^{N_t}_{ij}\right).
\]

## 5. Tesseract B — transport de fumée

### 5.1 Modèle continu

\[
\frac{\partial c}{\partial t}
+w\cdot\nabla c
=D_c\Delta c-\lambda_cc+S,
\]

avec

\[
D_c>0,\qquad \lambda_c\ge0,\qquad c(x,0)=0.
\]

### 5.2 Conditions aux limites

- advection : concentration extérieure nulle aux frontières d'entrée ;
- sortie : flux convectif sortant, sans réinjection ;
- diffusion : flux normal nul.

Le couple entrée/sortie dépend du signe de chaque composante de \(w\). Les mêmes
règles discrètes sont utilisées dans l'opérateur direct et dans sa transposée.

### 5.3 Schéma explicite

Pour un champ \(q\), définir

\[
D_x^-q_{ij}=\frac{q_{ij}-q_{i-1,j}}{\Delta x},
\qquad
D_x^+q_{ij}=\frac{q_{i+1,j}-q_{ij}}{\Delta x},
\]

et de même en \(y\). L'advection upwind vaut

\[
\mathcal L_{\mathrm{adv}}(w)c
=
-w_x^+D_x^-c-w_x^-D_x^+c
-w_y^+D_y^-c-w_y^-D_y^+c.
\]

Le Laplacien centré est noté \(\mathcal L_\Delta\). Une étape s'écrit

\[
c^{n+1}=A(\psi)c^n+\Delta t\,S^n,
\]

\[
A(\psi)
=I+\Delta t
\left[
\mathcal L_{\mathrm{adv}}(w)
+D_c\mathcal L_\Delta
-\lambda_cI
\right],
\qquad
\psi=(w_x,w_y,D_c,\lambda_c).
\]

### 5.4 CFL

La condition suffisante imposée est

\[
\nu_c=
\Delta t\left[
\frac{|w_x|}{\Delta x}
+\frac{|w_y|}{\Delta y}
+2D_c\left(
\frac1{\Delta x^2}+\frac1{\Delta y^2}
\right)
+\lambda_c
\right]\le1.
\]

Le solveur doit échouer explicitement avant la première étape si
\(\nu_c>1\), \(D_c\le0\), \(\lambda_c<0\), si la forme de \(S\) est invalide ou
si une position de capteur est hors domaine.

### 5.5 Observation aux capteurs

Pour chaque capteur fixe \(x_j\), \(H_j\) est l'opérateur d'interpolation
bilinéaire sur les quatre cellules voisines. Les prédictions sont

\[
\widehat y_{j,n}=H_jc^n+b_j.
\]

Les données synthétiques sont

\[
y_{j,n}
=\widehat y_{j,n}(\theta_{\mathrm{true}})
+\epsilon_{j,n},
\qquad
\epsilon_{j,n}\sim\mathcal N(0,\sigma_j^2).
\]

Dans Tiny, \(b_j=0\), \(\sigma_j\) est connu, la graine est fixe et le masque
d'observation est généré une seule fois puis réutilisé pour toutes les
évaluations.

## 6. Adjoint discret du transport

Le VJP doit être celui du programme discret, frontières et interpolation
incluses.

Soit \(\bar y^n\) le cotangent des sorties capteurs. On injecte d'abord

\[
\bar c^n \mathrel{+}=H^\top\bar y^n.
\]

Puis, pour \(n=N_t-1,\ldots,0\),

\[
\bar S^n\mathrel{+}=\Delta t\,\bar c^{n+1},
\]

\[
\bar\psi_k\mathrel{+}
=
(\bar c^{n+1})^\top
\left[
\frac{\partial A(\psi)}{\partial\psi_k}
\right]c^n,
\]

\[
\bar c^n\mathrel{+}=A(\psi)^\top\bar c^{n+1}.
\]

Les dérivées simples incluent

\[
\frac{\partial A}{\partial D_c}
=\Delta t\,\mathcal L_\Delta,
\qquad
\frac{\partial A}{\partial\lambda_c}
=-\Delta t\,I.
\]

Pour le vent, la dérivée utilise la branche upwind réellement active. Tiny
garantit que \(w_x\) et \(w_y\) ne changent pas de signe dans le voisinage du
gradient check.

Le VJP est calculé en coût
\(\mathcal O(N_tN_xN_y)\), indépendamment du nombre de composantes de \(S\)
différentiées. Une différence finie par composante de \(S\) est interdite.

Le Tesseract est sans état entre endpoints. L'endpoint VJP peut donc rejouer le
direct pour reconstruire \(c^0,\ldots,c^{N_t}\), puis effectuer la passe
inverse. Pour Tiny, ce stockage représente seulement
\((N_t+1)N_xN_y\) scalaires.

La validation unitaire obligatoire est

\[
\frac{
\left|
\langle Jv,q\rangle-\langle v,J^\top q\rangle
\right|
}{
\max\left(
1,|\langle Jv,q\rangle|,|\langle v,J^\top q\rangle|
\right)
}
<10^{-6}
\]

en double précision sur un petit cas déterministe.

## 7. Composition et gradient de bout en bout

L'orchestrateur JAX appelle les deux Tesseracts à l'intérieur de la fonction de
loss :

\[
(S,\ldots)=f_{\mathrm{fire}}(\theta),
\qquad
\widehat y=g_{\mathrm{smoke}}(S,w(\delta_\phi)).
\]

Pour \(\bar y=\partial\mathcal L/\partial\widehat y\), le Tesseract fumée
retourne

\[
\bar S=
\left(\frac{\partial g}{\partial S}\right)^\top\bar y,
\qquad
\bar w=
\left(\frac{\partial g}{\partial w}\right)^\top\bar y.
\]

Le Tesseract feu utilise l'autodiff PyTorch pour calculer

\[
\left(\frac{\partial f_{\mathrm{fire}}}{\partial\theta}\right)^\top
\bar S.
\]

La composante angulaire reçoit aussi

\[
\frac{\partial\mathcal L}{\partial\delta_\phi}
\mathrel{+}=
\bar v^\top\frac{\partial v}{\partial\delta_\phi}
+\bar w^\top\frac{\partial w}{\partial\delta_\phi}.
\]

Aucun fichier intermédiaire n'est utilisé par la fonction différentiée.

## 8. Vraisemblance et MAP

Soit \(m_{j,n}\in\{0,1\}\) le masque fixe et
\(N_{\mathrm{obs}}=\sum m_{j,n}\). La loss de données est

\[
\mathcal L_{\mathrm{data}}(\theta)
=
\frac1{N_{\mathrm{obs}}}
\sum_{j,n}m_{j,n}
\left[
\frac12
\left(
\frac{\widehat y_{j,n}(\theta)-y_{j,n}}{\sigma_j}
\right)^2
+\frac12\log(2\pi\sigma_j^2)
\right].
\]

Priors du premier cas :

\[
x_0,y_0\sim\operatorname{Uniform}
(m_x,1-m_x),
\]

\[
\log A_0\sim\mathcal N(\mu_A,s_A^2),
\qquad
\delta_\phi\sim
\mathcal N(0,s_\phi^2)
\text{ tronquée à }[-\delta_{\max},\delta_{\max}].
\]

La loss MAP est le négatif scalaire du log-posterior :

\[
\mathcal L_{\mathrm{MAP}}
=\mathcal L_{\mathrm{data}}
+\frac{(\log A_0-\mu_A)^2}{2s_A^2}
+\frac{\delta_\phi^2}{2s_\phi^2}
+C,
\]

à l'intérieur des bornes. Sa forme JAX doit être un scalaire de rang zéro.

L'optimisation Tiny utilise un vrai \(\operatorname{value\_and\_grad}\) de cette
loss, 10 à 30 itérations, une initialisation volontairement erronée et aucune
SVI.

## 9. Gradient check

Pour chaque composante physique \(\theta_k\),

\[
g_k^{\mathrm{FD}}(h)
=
\frac{
\mathcal L(\theta+he_k)
-\mathcal L(\theta-he_k)
}{2h}.
\]

Le pas nominal est

\[
h_k=\varepsilon_{\mathrm{FD}}
\max(1,|\theta_k|),
\]

et la sensibilité est évaluée à
\(h_k/2,h_k,2h_k\).

L'erreur relative rapportée est

\[
e_k^{\mathrm{rel}}
=
\frac{|g_k^{\mathrm{Tess}}-g_k^{\mathrm{FD}}|}
{\max(10^{-12},|g_k^{\mathrm{Tess}}|+|g_k^{\mathrm{FD}}|)}.
\]

Le signe est déclaré cohérent si le produit est positif, ou si les deux
gradients sont inférieurs à un seuil absolu documenté.

Critères :

- aucune valeur NaN ou infinie ;
- erreur relative médiane visée \(<10^{-2}\) ;
- erreur relative médiane acceptable au premier MVP \(<5\times10^{-2}\) ;
- conditions, observations, masque et graine identiques ;
- temps séparés pour VJP Tesseract et différences finies.

## 10. Exposition sanitaire

Pour chaque zone \(r\),

\[
N_r=\int_\Omega p_r(x)\,dx
\approx\Delta x\Delta y\sum_{ij}p_{r,ij},
\qquad N_r>0.
\]

La dose moyenne par personne est

\[
e_r
=
\frac1{N_r}
\int_0^{T_f}\int_\Omega
c(x,t)p_r(x)
\left[1-\eta_{\mathrm{filter},r}u_{\mathrm{filter},r}\right]
\,dx\,dt,
\]

soit discrètement

\[
e_r
=
\frac{\Delta t\Delta x\Delta y}{N_r}
\sum_{n,i,j}
c^n_{ij}p_{r,ij}
\left[1-\eta_{\mathrm{filter},r}u_{\mathrm{filter},r}\right].
\]

L'impact sanitaire incrémental est

\[
\Delta H_r
=
N_r
\left[
\operatorname{softplus}(a_r+b_re_r)
-\operatorname{softplus}(a_r)
\right],
\qquad b_r\ge0.
\]

Ainsi, \(e_r=0\Rightarrow\Delta H_r=0\). Les coefficients sont synthétiques et
ne doivent jamais être présentés comme prédictions cliniques.

## 11. Pertes financières

La perte physique ou sociétale est

\[
L_{\mathrm{phys}}
=
c_H\sum_r\Delta H_r
+c_BA_{\mathrm{burned}}
+c_DD_{\mathrm{interruption}}.
\]

\(D_{\mathrm{interruption}}\) est un indicateur synthétique différentiable ou
une donnée fixée ; sa définition exacte doit être dans la configuration.

Un paiement paramétrique lissé est

\[
\mathrm{Payout}
=
u_{\mathrm{insurance}}C_{\mathrm{cover}}
\sigma\left(
\frac{I_{\mathrm{event}}-\tau_{\mathrm{trigger}}}
{\varepsilon_{\mathrm{trigger}}}
\right),
\]

et la prime

\[
\mathrm{Premium}
=\pi u_{\mathrm{insurance}}C_{\mathrm{cover}}.
\]

La perte nette portée par l'organisation est

\[
L_{\mathrm{net}}
=
L_{\mathrm{phys}}
-\mathrm{Payout}
+\mathrm{Premium}
+C_{\mathrm{prevention}}
+C_{\mathrm{filtering}}.
\]

Le besoin de liquidité résiduel est

\[
L_{\mathrm{liquidity}}
=
\operatorname{smoothplus}_\tau
\left(
L_{\mathrm{phys}}-\mathrm{Payout}-\mathrm{Reserve}
\right),
\]

avec

\[
\operatorname{smoothplus}_\tau(x)
=\tau\log\left(1+e^{x/\tau}\right),
\]

évaluée par une formulation numériquement stable.

L'assurance ne figure dans aucune équation de \(T\), \(F\), \(S\), \(c\) ou
\(\Delta H\). Elle ne peut donc réduire ni le feu, ni la fumée, ni les impacts
sanitaires.

## 12. CVaR différentiable

Pour des pertes \(L_s\) et \(\alpha\in(0,1)\),

\[
\operatorname{CVaR}_\alpha(L)
\approx
\min_\zeta
\left[
\zeta+
\frac1{(1-\alpha)S}
\sum_{s=1}^S
\operatorname{smoothplus}_\tau(L_s-\zeta)
\right].
\]

Aucun tri de scénarios n'est autorisé dans la boucle de gradient.

Pour des échantillons postérieurs fixes \(\theta_s\),

\[
\begin{aligned}
J(u,\zeta)
&=
\frac1S\sum_s
\left[
L_{\mathrm{phys}}(u,\theta_s)
+C_{\mathrm{investment}}(u)
\right]\\
&\quad+
\gamma
\left[
\zeta+
\frac1{(1-\alpha)S}
\sum_s
\operatorname{smoothplus}_\tau
\left(
L_{\mathrm{net}}(u,\theta_s)-\zeta
\right)
\right]\\
&\quad+\mu\,\mathrm{Equity}(u).
\end{aligned}
\]

L'optimisation complète de \(J\) est hors C0 ; seuls la fonction et ses tests
d'invariants sont requis.

## 13. Budget différentiable

Pour des variables libres \(z_i\),

\[
\mathrm{share}_i
=
\frac{e^{z_i}}{\sum_ke^{z_k}},
\qquad
\mathrm{budget}_i=B\,\mathrm{share}_i.
\]

Les intensités physiques sont

\[
u_i=1-\exp
\left(
-\frac{\mathrm{budget}_i}{\mathrm{scale}_i}
\right).
\]

Cette paramétrisation garantit, à la précision machine :

\[
\mathrm{budget}_i\ge0,\qquad
\sum_i\mathrm{budget}_i=B,\qquad
0\le u_i<1.
\]

## 14. Identifiabilité et UQ

Risques principaux :

1. \(A_0\) et \(\eta_{\mathrm{smoke}}\) agissent tous deux sur l'amplitude ;
2. \((x_0,y_0)\) et \(\delta_\phi\) peuvent se compenser avec des capteurs mal
   placés ;
3. \(D_c\), \(\lambda_c\) et l'amplitude deviennent corrélés sur une fenêtre
   temporelle trop courte ;
4. des capteurs colinéaires ou uniquement amont dégradent fortement la
   localisation ;
5. un clamp ou un changement de signe upwind rend le gradient local peu
   informatif.

Mesures C0 :

- trois capteurs non colinéaires, dont au moins deux à l'aval ;
- \(D_T,D_c,\eta_{\mathrm{smoke}},\lambda_c,M,s_T,s_c\) connus ;
- vérité et point initial strictement intérieurs ;
- plusieurs instants observés ;
- comparaison à différences finies sur le pipeline complet.

Ordre obligatoire après validation du MAP :

1. MAP ;
2. Hessienne et approximation de Laplace autour du MAP ;
3. SVI NumPyro ;
4. NUTS seulement sur une expérience réduite.

## 15. Cas Tiny proposé

Un seul fichier de configuration contiendra :

| Élément | Valeur proposée |
|---|---|
| domaine | \([0,1]^2\) |
| grille | \(32\times32\), cellules centrées |
| pas de temps | 40 |
| capteurs | 3 positions fixes, non colinéaires et intérieures |
| ignition | une gaussienne continue |
| vent | constant, composantes non nulles |
| combustible | uniforme, sans prévention |
| humidité | uniforme et fixée |
| biais capteurs | zéro et connu |
| bruit | gaussien, écart-type connu |
| observations masquées | 5 %, masque fixe |
| graine | 20260805 |
| données publiques | aucune |

Les valeurs numériques des coefficients et de \(\Delta t\) seront figées dans
le fichier seulement après vérification simultanée de \(\nu_T\le1\) et
\(\nu_c\le1\). Elles seront ensuite immuables pour les trois commandes Tiny.

## 16. Tests mathématiques obligatoires

### FireSpread

- \(F=0\Rightarrow R=0\) ;
- humidité accrue \(\Rightarrow R\) réduite à \(T,F\) fixés ;
- \(F^{n+1}\le F^n\) et \(F\in[0,1]\) ;
- sans réaction, \(T\) nul reste nul et une intensité positive décroît ;
- symétrie de la diffusion sans vent ;
- déplacement cohérent avec le vent ;
- sensibilité continue à \(x_0,y_0\) ;
- échec si CFL invalide ;
- stabilité et absence de saturation sur Tiny.

### SmokeTransport

- \(S=0\Rightarrow c=0\) ;
- masse diminuée lorsque \(\lambda_c>0\) sans source ;
- transport dans le sens du vent ;
- étalement accru avec \(D_c\) ;
- interpolation bilinéaire exacte sur un champ affine ;
- échec si CFL invalide ;
- test de produit scalaire de l'adjoint.

### Pipeline

- schémas valides ;
- déterminisme à graine fixe ;
- loss scalaire ;
- gradient fini ;
- gradient check et sensibilité au pas ;
- diminution de la loss MAP ;
- amélioration de la distance de \(x_0,y_0\) à la vérité ;
- aucun fichier intermédiaire implicite.

### Santé-finance

- \(e_r=0\Rightarrow\Delta H_r=0\) ;
- filtration accrue ne peut augmenter \(e_r\) ;
- assurance sans effet sur les champs physiques ;
- prévention peut réduire \(L_{\mathrm{phys}}\) ;
- CVaR supérieure ou égale à la perte moyenne sur les scénarios de queue
  appropriés, à la tolérance de lissage documentée ;
- budget exactement respecté par softmax.

## 17. Limites d'usage

- modèle adimensionné et maillage très grossier ;
- coefficients de réaction, santé et finance synthétiques ;
- vent constant, topographie et humidité fixes ;
- aucune chimie des aérosols, flottabilité, dépôt sec ou verticalité ;
- conditions aux limites simplifiées ;
- schéma upwind du premier ordre, diffusif numériquement ;
- trois capteurs seulement ;
- bruit homoscédastique connu dans le premier cas ;
- aucune calibration à des données réelles ;
- aucune validité clinique, réglementaire, assurantielle ou actuarielle ;
- aucune décision financière complète dans C0.

## 18. Décisions soumises à validation

La validation demandée porte explicitement sur ces six choix :

1. même grille \(32\times32\) et même \(\Delta t\) pour les deux solveurs ;
2. vent angulaire partagé par FireSpread et SmokeTransport ;
3. absence de clamps dans le chemin nominal, remplacés par CFL et contraintes
   de positivité ;
4. frontières : flux diffusif nul, entrée advective homogène et sortie
   convective ;
5. loss MAP fondée sur les capteurs et les priors ; santé-finance en diagnostics
   JAX séparés pendant C0 ;
6. biais capteurs nuls connus et écart-type du bruit connu dans Tiny.
