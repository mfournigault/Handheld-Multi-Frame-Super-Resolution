# Analyse des problèmes de performance — `Handheld-Multi-Frame-Super-Resolution`

> **Dépôt analysé :** `mfournigault/Handheld-Multi-Frame-Super-Resolution`  
> **Branche :** `main`  
> **Date :** Mai 2026

---

## Table des matières

1. [Contexte et cadre de comparaison](#1-contexte-et-cadre-de-comparaison)
2. [Points problématiques majeurs](#2-points-problématiques-majeurs)
   - [2.1 Compilation JIT Numba — surcoût cold-start massif](#21-compilation-jit-numba--surcoût-cold-start-massif)
   - [2.2 Coexistence PyTorch + Numba CUDA et synchronisations implicites](#22-coexistence-pytorch--numba-cuda-et-synchronisations-implicites)
   - [2.3 Boucle Python par image avec synchronisations de timing](#23-boucle-python-par-image-avec-synchronisations-de-timing)
   - [2.4 Double transfert CPU→GPU pour les images grise des frames comparées](#24-double-transfert-cpugpu-pour-les-images-grises-des-frames-comparées)
   - [2.5 Calcul du modèle de bruit Monte-Carlo à chaque exécution](#25-calcul-du-modèle-de-bruit-monte-carlo-à-chaque-exécution)
   - [2.6 Réallocation de kernels de gradient à chaque frame](#26-réallocation-de-kernels-de-gradient-à-chaque-frame)
   - [2.7 `extract_flow_patches` — allocation de grand tenseur intermédiaire](#27-extract_flow_patches--allocation-de-grand-tenseur-intermédiaire)
   - [2.8 Bubble sort dans un kernel CUDA](#28-bubble-sort-dans-un-kernel-cuda)
   - [2.9 Configuration de threads sous-optimale pour certains kernels](#29-configuration-de-threads-sous-optimale-pour-certains-kernels)
   - [2.10 Synchronisation dans la boucle ICA](#210-synchronisation-dans-la-boucle-ica)
   - [2.11 Bug logique dans le block matching L1](#211-bug-logique-dans-le-block-matching-l1)
   - [2.12 Warp 0 non écrit en mémoire partagée dans ICA kernel 64](#212-warp-0-non-écrit-en-mémoire-partagée-dans-ica-kernel-64)
3. [Comparaison conceptuelle avec l'implémentation mobile de Google](#3-comparaison-conceptuelle-avec-limplémentation-mobile-de-google)
4. [Fichiers et fonctions responsables des lenteurs](#4-fichiers-et-fonctions-responsables-des-lenteurs)
5. [Suggestions d'amélioration priorisées](#5-suggestions-damélioration-priorisées)
6. [Résumé des gains attendus](#6-résumé-des-gains-attendus)

---

## 1. Contexte et cadre de comparaison

L'algorithme original de Wronski et al. (Google Pixel 3) tourne sur un **ISP (Image Signal Processor) matériel dédié**, intégré au SoC Snapdragon/Pixel. Ces unités traitent les données directement depuis le capteur, en pipeline, avec des latences sub-milliseconde par frame et une consommation en milliwatts.

Cette implémentation Python tourne, elle, sur un **GPU desktop via Numba + PyTorch**, deux frameworks aux modèles d'exécution distincts et soumis à l'overhead Python. Le README l'assume explicitement :

> *"not as fast as that of Google. It is mainly for scientific and educational purpose"*

et mentionne **4 secondes pour 20 images 12MP** sur une RTX 3090 *sans compter la compilation JIT de Numba*.

L'objectif de ce document est d'identifier précisément les sources de lenteur, de les prioriser, et de proposer des actions concrètes pour améliorer significativement les performances — sans pour autant prétendre rivaliser avec un ISP matériel dédié.

---

## 2. Points problématiques majeurs

### 2.1 Compilation JIT Numba — surcoût cold-start massif

**Fichiers concernés :** `merge.py`, `robustness.py`, `ICA.py`, `block_matching.py`, `kernels.py`, `utils_image.py`, `utils.py`

Numba compile ses kernels CUDA en PTX lors du **premier appel** de chaque kernel. Cette compilation peut prendre plusieurs secondes pour des kernels complexes :

- `accumulate` (`merge.py`) — ~150 lignes
- `ica_kernel_64` (`ICA.py`) — ~120 lignes
- `cuda_L1_local_search64` (`block_matching.py`) — ~90 lignes

Aucun mécanisme de cache de compilation AOT (ahead-of-time) n'est en place. Chaque démarrage d'un nouveau processus Python déclenche une recompilation complète de tous les kernels utilisés.

> **Impact estimé :** 10 à 60 secondes de surcoût au premier run. Rédhibitoire pour tout déploiement industriel ou serveur.

---

### 2.2 Coexistence PyTorch + Numba CUDA et synchronisations implicites

**Fichiers concernés :**
- `alignment.py` — lignes 36, 87, 107
- `ICA.py` — lignes 23–33
- `kernels.py` — lignes 94–116
- `block_matching.py` — lignes 353–354

La codebase mélange deux backends GPU incompatibles :

| Backend | Utilisé pour |
|---|---|
| **PyTorch** | Pyramides gaussiennes, FFT, convolutions ICA, block matching L2 |
| **Numba CUDA** | Merge, robustesse, GAT, kernels ICA et L1 |

Les conversions entre les deux nécessitent des **synchronisations GPU explicites**. Exemple dans `alignment.py` à la ligne 107 :

```python
# alignment.py:107
cuda.synchronize()  # PyTorch could read 'alignment' before the kernel writes finish
```

Ce `cuda.synchronize()` se trouve dans la boucle principale des niveaux pyramidaux, exécutée **pour chaque image du burst**. Il force le pipeline GPU à vider toutes les opérations en attente avant de continuer, détruisant toute possibilité de **pipelining GPU**.

Les conversions `cuda.as_cuda_array(gradx)` (`ICA.py:23`) et `torch.as_tensor(img_grey, ...)` (`kernels.py:94`) impliquent également des synchronisations implicites si les opérations précédentes ne sont pas terminées.

> **Impact estimé :** Pour 20 images et 4 niveaux pyramidaux = **80 barrières de synchronisation GPU** dans la boucle principale.

---

### 2.3 Boucle Python par image avec synchronisations de timing

**Fichier :** `super_resolution.py` — lignes 133–173

```python
# super_resolution.py:133-173
for im_id in range(n_images):
    if verbose:
        cuda.synchronize()               # ligne 135
    cuda.to_device(comp_imgs[im_id], ...)
    cuda_im_grey = compute_grey_images(comp_imgs[im_id], grey_method)
    cuda_final_alignment = align_(...)
    cuda_robustness = compute_robustness_(...)
    cuda_kernels = estimate_kernels_(...)
    merge_(...)
    if verbose:
        cuda.synchronize()               # ligne 169
    stream.synchronize()                 # ligne 173
```

Deux problèmes majeurs :

1. La ligne `stream.synchronize()` (L173) attend la fin de **tous les kernels GPU** avant de commencer l'image suivante — il est impossible de chevaucher le transfert de l'image `n+1` pendant le traitement de l'image `n`.
2. Le `cuda.synchronize()` à la ligne 135 s'exécute même à `verbose=1` (activé par défaut dans `configs/default.yaml`).

> **Impact estimé :** Pipeline GPU 100 % séquentiel, aucun overlapping CPU/GPU ou inter-images possible.

---

### 2.4 Double transfert CPU→GPU pour les images grises des frames comparées

**Fichier :** `super_resolution.py` — lignes 141–147

```python
# super_resolution.py:141-147
cuda.to_device(comp_imgs[im_id], to=cuda_img, stream=stream)  # Transfert 1 (GPU)

if bayer_mode:
    cuda_im_grey = compute_grey_images(comp_imgs[im_id], grey_method)  # comp_imgs[im_id] = CPU !
```

`comp_imgs[im_id]` est un tableau **NumPy (CPU)**, mais `cuda_img` (GPU) contient déjà les mêmes données. La fonction `compute_grey_images` avec `method="FFT"` (`utils_image.py:83`) appelle :

```python
# utils_image.py:83
torch_img_grey = th.as_tensor(img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")  # Transfert 2 !
```

Ce second transfert PCIe est **entièrement redondant**. De plus, la méthode FFT enchaîne 4 transformées complètes (FFT2 → fftshift → masquage → ifftshift → IFFT2) pour obtenir une simple image en niveaux de gris, alors que la méthode `"decimating"` (moyennage 2×2 quads) obtient un résultat équivalent pour l'alignement en O(N).

> **Impact estimé :** Pour 20 images de 12MP (4000×3000 float32 = 48 MB chacune) : ~960 MB de transferts PCIe superflus + surcoût FFT majeur.

---

### 2.5 Calcul du modèle de bruit Monte-Carlo à chaque exécution

**Fichier :** `super_resolution.py` — ligne 254 ; `fast_monte_carlo.py`

```python
# super_resolution.py:254
std_curve, diff_curve = run_fast_MC(alpha, beta)
```

`run_fast_MC` lance un multiprocessing Monte Carlo sur `n_patches = 1e5` patchs × `n_brightness_levels = 1000` niveaux de luminosité (`fast_monte_carlo.py:21-22`). Ce calcul est relancé **à chaque appel à `process()`**, alors que les courbes dépendent uniquement de `alpha` et `beta`.

Ironiquement, le code commenté aux lignes 247–251 de `super_resolution.py` montrait un mécanisme de chargement depuis des fichiers `.npy` précalculés, qui a été **remplacé** par le calcul à la volée.

> **Impact estimé :** 2 à 10 secondes de surcoût CPU pour chaque appel de pipeline, pendant lesquelles le GPU reste inactif.

---

### 2.6 Réallocation de kernels de gradient à chaque frame

**Fichier :** `kernels.py` — lignes 97–116

```python
# kernels.py:97-116 — appelé pour CHAQUE image
grad_kernel1 = np.array([[[[-0.5, 0.5]]], [[[ 0.5, 0.5]]]]])
grad_kernel1 = th.as_tensor(grad_kernel1, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
grad_kernel2 = np.array([[[[0.5], [0.5]]], [[[-0.5], [0.5]]]]])
grad_kernel2 = th.as_tensor(grad_kernel2, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
```

Ces kernels sont **constants**, mais recréés à chaque appel à `estimate_kernels()`. Idem pour le kernel gaussien dans `cuda_downsample` (`utils_image.py:380`) :

```python
# utils_image.py:380
gaussian_kernel = _gaussian_kernel1d(sigma=factor * 0.5, ...)[::-1].copy()
th_gaussian_kernel = torch.as_tensor(gaussian_kernel, ..., device="cuda")
```

Ce calcul est effectué à chaque niveau de pyramide de chaque image. Pour 20 images à 4 niveaux = **80 appels inutiles**. La bonne pratique (déjà appliquée pour `SOBEL_X/Y` dans `alignment.py:15-18` et `ICA.py:10-11`) n'a pas été systématisée.

> **Impact estimé :** Allocation/désallocation GPU répétée, pression sur l'allocateur CUDA, overhead Python non négligeable.

---

### 2.7 `extract_flow_patches` — allocation de grand tenseur intermédiaire

**Fichier :** `block_matching.py` — lignes 348–377

```python
# block_matching.py:366-377
y_coords = top[:, :, None, None] + dy_offsets[None, None, :, :]
x_coords = left[:, :, None, None] + dx_offsets[None, None, :, :]
y_flat = y_coords.reshape(-1)
x_flat = x_coords.reshape(-1)
aligned_patches = frame_tgt[y_flat, x_flat].view(ny, nx, P_search, P_search)
```

Pour un niveau fin avec `tile_size=32`, `search_radius=4` et une image 12MP :

- `P_search = 2×4 + 32 = 40`
- Nombre de patches : `ny ≈ 93, nx ≈ 125`
- Taille du tenseur : `93 × 125 × 40 × 40 × 4 octets ≈ 74 MB`

L'indexation avancée `frame_tgt[y_flat, x_flat]` crée une **copie complète** des patches. Avec 4 niveaux × 20 images = **80 allocations de ~74 MB**, soit potentiellement des gigaoctets de mémoire GPU alloués/libérés.

> **Impact estimé :** Fragmentation mémoire GPU, pression sur l'allocateur, risque d'OOM sur GPU avec peu de VRAM.

---

### 2.8 Bubble sort dans un kernel CUDA

**Fichier :** `utils_image.py` — lignes 303–309

```python
# utils_image.py:303-309
@cuda.jit(device=True)
def bubble_sort(X):
    N = X.size
    for i in range(N-1):
        for j in range(N-i-1):
            if X[j] > X[j+1]:
                X[j], X[j+1] = X[j+1], X[j]
```

Utilisé dans `cuda_frame_count_denoising_median` (ligne 291). Bubble sort est **O(N²)** exécuté en série dans des registres locaux d'un kernel CUDA. Pour `radius=3` (fenêtre 7×7 = 49 éléments) : `49×48/2 = 1176` comparaisons par thread, 100 % sérielles.

Un tri par insertion ou bitonic sort serait bien plus adapté au contexte GPU.

> **Impact estimé :** Le post-processing median est ~10× plus lent que nécessaire si activé.

---

### 2.9 Configuration de threads sous-optimale pour certains kernels

**Fichier :** `utils.py` — ligne 23 ; `robustness.py` — ligne 258

```python
# utils.py:23
DEFAULT_THREADS = 16  # → 16×16 = 256 threads/bloc
```

```python
# robustness.py:258
threadsperblock = (1, DEFAULT_THREADS, DEFAULT_THREADS)  # compute_local_stats
```

Problèmes :

- Le maximum CUDA est **1024 threads/bloc**. Utiliser 256 laisse 75 % de la capacité inutilisée.
- La configuration `(1, 16, 16)` pour `compute_local_stats` signifie que la dimension canaux vaut 1 : 3 blocs distincts sont lancés pour les 3 canaux Bayer, chacun avec seulement 256 threads.
- L'occupancy GPU peut être augmentée significativement avec 32×32 = 1024 threads pour les kernels 2D.

> **Impact estimé :** Occupancy GPU limitée à ~25 % du maximum possible pour de nombreux kernels.

---

### 2.10 Synchronisation dans la boucle ICA

**Fichier :** `ICA.py` — ex. `ica_kernel_8`, lignes 141–179

```python
# ICA.py:141-179
for _ in range(niter):
    cuda.syncthreads()   # dans la boucle
    ...
    while N > 0:
        cuda.syncthreads()
        N = N // 2
```

L'ICA tourne `niter=3` fois (configuré dans `configs/default.yaml:18`). Chaque itération contient une `syncthreads()` initiale + `log2(TILE_SIZE²)` syncthreads pour la réduction. Pour `tile_size=16` : `3 × (1 + 7) = 24 syncthreads` par bloc. C'est algorithmiquement inévitable, mais la valeur de `niter` pourrait être réduite aux niveaux grossiers de la pyramide (1 itération suffit là où l'alignement initial est bon).

---

### 2.11 Bug logique dans le block matching L1

**Fichier :** `block_matching.py` — lignes 167–180 (`cuda_L1_local_search16`) et équivalents 32/64

```python
# block_matching.py:167-180
if tid == 0:
    err = s_err[0, 0]
    for i in range(2*search_radius + 1):
        for j in range(2*search_radius + 1):
            min = s_err[i, j]
            if err < min:          # ← condition INVERSÉE pour trouver un minimum
                min = err
                min_shift_y = i - search_radius
                min_shift_x = j - search_radius

alignments[tile_y, tile_x, 0] = s_flow[0] + min_shift_x  # ← hors du bloc if tid==0 !
alignments[tile_y, tile_x, 1] = s_flow[1] + min_shift_y
```

Deux bugs distincts :

1. **Condition inversée** : `if err < min` ne met jamais à jour `min_shift_y/x` correctement (il faudrait `if min < err`).
2. **Race condition** : L'écriture dans `alignments` se fait **hors du bloc `if tid == 0:`**, donc tous les threads du bloc écrivent simultanément avec `min_shift_x` non initialisé pour la plupart.

Ces bugs sont identiques dans `cuda_L1_local_search32` (lignes 240–252) et `cuda_L1_local_search64` (lignes 333–345).

> **Impact :** Le block matching L1 (utilisé au niveau le plus fin selon `metrics: ['L1', ...]` dans `default.yaml:17`) peut produire des alignements incorrects, dégradant la qualité ET forçant l'ICA à partir d'un mauvais point initial.

---

### 2.12 Warp 0 non écrit en mémoire partagée dans ICA kernel 64

**Fichier :** `ICA.py` — ligne 466

```python
# ICA.py:466
if ti % WARP_SIZE == 0 and ti > 0:  # Le warp 0 n'écrit pas dans s_B0[0]
    s_B0[ti//WARP_SIZE] = B0
    s_B1[ti//WARP_SIZE] = B1
```

Le warp 0 conserve `B0` dans un registre local et fait la somme finale directement. `s_B0[0]` reste **non initialisé**, ce qui est potentiellement risqué si la logique de sommation évolue.

---

## 3. Comparaison conceptuelle avec l'implémentation mobile de Google

| Aspect | Implémentation Python (ce repo) | ISP Google Pixel 3 |
|---|---|---|
| **Plateforme** | GPU desktop (RTX 3090) via Numba+PyTorch | ASP/ISP hardware dédié dans le SoC |
| **Compilation** | JIT à la première exécution (10–60s) | Circuits figés, pas de compilation |
| **Overhead Python** | Présent à chaque kernel launch | Zéro |
| **Synchronisation** | Explicite, fréquente (80+ par burst) | Pipeline hardware continu |
| **Transferts mémoire** | PCIe CPU↔GPU (potentiellement redondants) | On-chip, bande passante maximale |
| **Modèle bruit** | Monte Carlo recalculé à chaque run (~5s) | Courbes précalculées en ROM |
| **Parallélisme inter-frames** | Séquentiel (`stream.synchronize`) | Pipeline avec recouvrement capture/traitement |
| **Threads par bloc** | 256 (16×16), max théorique 1024 | N/A (hardware parallèle natif) |
| **Consommation** | ~350W (RTX 3090 TDP) | ~2W (ISP mobile) |
| **Latence** | ~4s pour 20 images 12MP | Sub-seconde en temps réel |

La différence de performance n'est pas seulement logicielle : l'ISP opère **in-situ** depuis les données capteur, sans jamais transiter par un bus PCIe, et exécute des pipelines câblés sans overhead d'interprétation ou de JIT. Cette implémentation Python est structurellement incapable d'atteindre ces performances, mais peut être significativement améliorée dans les limites de son architecture.

---

## 4. Fichiers et fonctions responsables des lenteurs

Classés par impact estimé sur le temps d'exécution total :

| Priorité | Fichier | Fonction / ligne | Cause principale |
|---|---|---|---|
| 🔴 Critique | `super_resolution.py` | `process()` L254 | Monte Carlo recalculé systématiquement |
| 🔴 Critique | `super_resolution.py` | `main()` boucle L133–173 | `stream.synchronize()` bloquant, pipeline séquentiel |
| 🔴 Critique | `alignment.py` | `align()` L107 | `cuda.synchronize()` dans la boucle pyramidale |
| 🟠 Majeur | `super_resolution.py` | `main()` L144–147 | Double transfert PCIe + FFT coûteuse pour greyscale |
| 🟠 Majeur | Tous les fichiers `@cuda.jit` | kernels | Compilation JIT cold-start |
| 🟠 Majeur | `block_matching.py` | `extract_flow_patches()` L348 | Allocation ~74 MB par frame par niveau |
| 🟡 Moyen | `kernels.py` | `estimate_kernels()` L97–116 | Réallocation de kernels constants |
| 🟡 Moyen | `utils_image.py` | `cuda_downsample()` L380 | Recompute kernel gaussien par appel |
| 🟡 Moyen | `utils.py` | `DEFAULT_THREADS = 16` L23 | Occupancy GPU limitée à ~25 % |
| 🟡 Moyen | `robustness.py` | `compute_local_stats()` L258 | Configuration `(1, 16, 16)` sous-optimale |
| 🟢 Faible | `utils_image.py` | `bubble_sort()` L303 | O(N²) en kernel GPU |
| 🟢 Faible | `block_matching.py` | `cuda_L1_local_search*` L167, 240, 333 | Bug logique + race condition |

---

## 5. Suggestions d'amélioration priorisées

### Quick wins — moins d'une journée chacun

#### QW-1 : Cache du modèle de bruit
**Fichier :** `super_resolution.py` L254

Ajouter un dictionnaire module-level pour mettre en cache les courbes calculées :

```python
_mc_cache = {}

def _get_noise_curves(alpha, beta):
    key = (round(alpha, 8), round(beta, 8))
    if key not in _mc_cache:
        _mc_cache[key] = run_fast_MC(alpha, beta)
    return _mc_cache[key]
```

Remplacer `run_fast_MC(alpha, beta)` par `_get_noise_curves(alpha, beta)`. Économise 2–10s à chaque appel après le premier.

#### QW-2 : Kernels de gradient constants
**Fichiers :** `kernels.py` L97–116, `utils_image.py` L380

Déplacer `grad_kernel1`, `grad_kernel2` et les kernels gaussiens précalculés comme variables de module, en suivant le pattern déjà appliqué pour `SOBEL_X/Y` dans `alignment.py:15-18`.

#### QW-3 : Corriger le grey image pour `comp_imgs`
**Fichier :** `super_resolution.py` L144–147

Passer `cuda_img` (GPU) plutôt que `comp_imgs[im_id]` (CPU) à `compute_grey_images` pour éliminer le second transfert PCIe.

#### QW-4 : Augmenter `DEFAULT_THREADS`
**Fichier :** `utils.py` L23

Passer à `DEFAULT_THREADS = 32` pour obtenir 32×32 = 1024 threads/bloc dans les kernels 2D (après vérification des contraintes de mémoire partagée de chaque kernel).

#### QW-5 : Corriger le bug L1 block matching
**Fichier :** `block_matching.py` L167–180 et équivalents

- Inverser la condition : `if min < err` au lieu de `if err < min`
- Déplacer les écritures dans `alignments` à l'intérieur du bloc `if tid == 0:`

---

### Optimisations moyen terme — 1 à 2 semaines

#### MT-1 : Éliminer les `cuda.synchronize()` de la boucle principale
**Fichier :** `alignment.py` L107

Remplacer par des CUDA Events (`torch.cuda.Event`) pour une synchronisation fine entre PyTorch et Numba, sans vider l'intégralité du pipeline GPU.

#### MT-2 : Remplacer la méthode FFT pour le greyscale des frames comparées
**Fichier :** `utils_image.py`, `super_resolution.py` L145

La méthode `"decimating"` (moyennage 2×2 quads) est suffisante pour l'alignement, où une légère aliasing n'affecte pas la précision du flot optique grossier. Elle remplace 4 FFT2D complètes par une simple moyenne, représentant un gain potentiel de 5–10× pour cette étape.

#### MT-3 : Supprimer `stream.synchronize()` dans la boucle
**Fichier :** `super_resolution.py` L173

Permettre le chevauchement des transferts de l'image `n+1` pendant le traitement de l'image `n`, via des streams CUDA distincts :

```python
stream_compute = cuda.stream()
stream_transfer = cuda.stream()
# Ping-pong : transfert de l'image n+1 pendant le calcul sur l'image n
```

#### MT-4 : Remplacer le bubble sort
**Fichier :** `utils_image.py` L303

Utiliser un tri par insertion (optimal pour les petits tableaux en kernel GPU) ou un bitonic sort pour des fenêtres plus grandes :

```python
@cuda.jit(device=True)
def insertion_sort(X):
    N = X.size
    for i in range(1, N):
        key = X[i]
        j = i - 1
        while j >= 0 and X[j] > key:
            X[j + 1] = X[j]
            j -= 1
        X[j + 1] = key
```

#### MT-5 : Réactiver le cache de courbes de bruit sur disque
**Fichier :** `super_resolution.py` L247–251 (code commenté)

Le mécanisme original de chargement depuis des fichiers `.npy` (actuellement commenté) est plus robuste que le cache mémoire (QW-1). Le réactiver en ajoutant le calcul à la demande et la mise en cache automatique.

#### MT-6 : Activer le cache de compilation Numba
**Tous les fichiers `@cuda.jit`**

Ajouter `@cuda.jit(cache=True)` aux kernels les plus lourds pour activer la sauvegarde sur disque du PTX compilé et éliminer le JIT cold-start après la première exécution.

---

### Optimisations structurelles — plus d'une semaine

#### ST-1 : Migration complète vers un seul backend GPU
Le mélange PyTorch/Numba est la source principale des synchronisations implicites. Deux options :

- **Option A (préférée)** : Migrer le merge et la robustesse vers PyTorch (opérations custom via extensions C++ CUDA ou `torch.compile`). Cela permet d'utiliser les streams PyTorch uniformément et d'exploiter les optimisations de graph.
- **Option B** : Migrer les pyramides et les FFT vers Numba pur (plus verbeux mais unifié).

#### ST-2 : Batching inter-images pour le merge
Le merge et la robustesse sont des opérations indépendantes par pixel de sortie. En empilant les `comp_imgs` sur un batch PyTorch, on réduit drastiquement les lancements de kernels et améliore l'utilisation du GPU.

#### ST-3 : `niter` variable par niveau pyramidal
Configurer `n_iter` différemment par niveau pyramidal dans `default.yaml`. Les niveaux grossiers n'ont pas besoin de 3 itérations ICA — 1 ou 2 suffisent, réduisant les syncthreads et le temps de calcul.

```yaml
ica:
  tuning:
    n_iter: [3, 2, 1, 1]  # fine-to-coarse
```

---

### Instrumentation et profiling à ajouter

#### PROF-1 : Utiliser NVIDIA Nsight Systems
```bash
nsys profile --trace=cuda,nvtx python run_handheld.py --impath test_burst --outpath /tmp/out.png
```
Visualiser la timeline GPU/CPU, identifier les gaps (GPU idle) et les synchronisations superflues.

#### PROF-2 : Utiliser NVIDIA Nsight Compute
```bash
ncu --set full python -c "from handheld_super_resolution import process; ..."
```
Mesurer l'occupancy, les memory transactions et l'efficacité des warps pour les kernels `accumulate`, `ica_kernel_32/64`, `cuda_L1_local_search*`.

#### PROF-3 : Ajouter des CUDA Events autour de chaque étape
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... kernel ...
end.record()
torch.cuda.synchronize()
print(f"Merge: {start.elapsed_time(end):.2f} ms")
```
Mesure les temps GPU réels sans synchronisation CPU coûteuse, complémentaire au mécanisme `timer` existant dans `utils.py`.

#### PROF-4 : Activer `verbose=3`
Le mécanisme `timer` dans `utils.py:128-146` est déjà implémenté — le passer à `verbose=3` donne un détail complet des temps par étape sans modification de code.

---

## 6. Résumé des gains attendus

| Optimisation | Gain estimé sur le temps total |
|---|---|
| QW-1 : Cache Monte Carlo | −2 à −10s (surcoût CPU éliminé) |
| QW-3 : Fix double transfert grey | −20 % par frame en mode Bayer+FFT |
| MT-1 : Suppression synchros inter-framework | −30 % sur la latence d'alignement |
| MT-6 / Compilation AOT | −10 à −60s cold-start éliminé |
| MT-3 : Stream overlap inter-images | −15 à −25 % sur le temps total burst |
| QW-4 : Threads 32×32 | +2 à 4× occupancy (gain variable par kernel) |
| QW-5 : Fix bug L1 | Qualité + convergence ICA améliorée |

**Gain total estimé** avec les quick wins + optimisations moyen terme : réduction du temps de traitement d'un burst 12MP×20 de ~4s à **< 1.5s sur RTX 3090**, et élimination du cold-start JIT.

---

*Rapport généré à partir de l'analyse statique du code source sur la branche `main`. Aucune modification de code n'a été effectuée.*
