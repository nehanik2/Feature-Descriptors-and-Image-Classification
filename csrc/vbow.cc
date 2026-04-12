#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <585/common/types.h>
#include <585/grad/grad.h>      // ivc::sobel_angle
#include <585/vbow/vbow.h>      // ivc::LogReg

#include "vbow/vbow.h"

namespace ivc
{
namespace student
{

// ============================================================
//  Helpers
// ============================================================

// Extract a square patch of side `patch_size` centred at (row, col)
// from a zero-padded image stored as a flat row-major byte vector.
// `img_rows` and `img_cols` are the ORIGINAL (unpadded) dimensions.
static ivc::GrayscaleByteImg extract_patch(const ivc::GrayscaleByteImg& img,
                                            int img_rows,
                                            int img_cols,
                                            int center_row,
                                            int center_col,
                                            int patch_size)
{
    int half = patch_size / 2;
    ivc::GrayscaleByteImg patch(patch_size * patch_size);

    for (int dr = 0; dr < patch_size; ++dr)
    {
        for (int dc = 0; dc < patch_size; ++dc)
        {
            int r = center_row - half + dr;
            int c = center_col - half + dc;
            if (r >= 0 && r < img_rows && c >= 0 && c < img_cols)
                patch[dr * patch_size + dc] = img[r * img_cols + c];
            else
                patch[dr * patch_size + dc] = 0; // zero-pad
        }
    }
    return patch;
}

Vocab::Vocab()
    : _patch_to_idx(), _idx_to_patch()
{}

void Vocab::add(const ivc::GrayscaleByteImg& e)
{
    if (_patch_to_idx.find(e) == _patch_to_idx.end())
    {
        uint64_t idx = static_cast<uint64_t>(_patch_to_idx.size());
        _patch_to_idx[e] = idx;
        _idx_to_patch[idx] = e;
    }
}

const size_t Vocab::size() const
{
    return _patch_to_idx.size();
}

// _idx_to_patch is std::map<uint64_t,...> so iteration is already in index order
std::list<ivc::GrayscaleByteImg> Vocab::ordered_elements() const
{
    std::list<ivc::GrayscaleByteImg> elements;
    // _idx_to_patch is a std::map keyed by uint64_t → already sorted ascending
    for (const auto& kv : _idx_to_patch)
        elements.push_back(kv.second);
    return elements;
}

// ============================================================
//  BagOfWords (base)
// ============================================================

BagOfWords::BagOfWords(const ivc::ByteDataset& X,
                       const size_t patch_size)
    : _vocab(), _patch_size(patch_size)
{
    if (X.empty()) return;

    // Infer per-image dimensions.
    // ivc::ByteDataset is std::vector<ivc::GrayscaleByteImg>
    // Each row is one flattened image; assume square images.
    const size_t img_pixels = X[0].size();
    const int    img_side   = static_cast<int>(std::round(std::sqrt(static_cast<double>(img_pixels))));

    const int ps  = static_cast<int>(_patch_size);

    for (const auto& img : X)
    {
        for (int r = 0; r < img_side; ++r)
        {
            for (int c = 0; c < img_side; ++c)
            {
                ivc::GrayscaleByteImg patch =
                    extract_patch(img, img_side, img_side, r, c, ps);
                _vocab.add(patch);
            }
        }
    }
}

// ============================================================
//  BinaryBagOfWords
// ============================================================

BinaryBagOfWords::BinaryBagOfWords(const ivc::ByteDataset& X,
                                   const size_t patch_size)
    : BagOfWords(X, patch_size)
{}

ivc::FloatDataset BinaryBagOfWords::transform(const ivc::ByteDataset& X) const
{
    const size_t V = _vocab.size();
    ivc::FloatDataset D(X.size(), ivc::FloatRow(V, 0.0f));

    if (X.empty() || V == 0) return D;

    // Build a local lookup: patch -> index
    // ordered_elements() returns patches in index order
    std::map<ivc::GrayscaleByteImg, uint64_t, GrayscaleByteImg_comp_t> lookup;
    uint64_t idx = 0;
    for (const auto& p : _vocab.ordered_elements())
        lookup[p] = idx++;

    const int img_pixels = static_cast<int>(X[0].size());
    const int img_side   = static_cast<int>(std::round(std::sqrt(static_cast<double>(img_pixels))));
    const int ps         = static_cast<int>(_patch_size);

    for (size_t i = 0; i < X.size(); ++i)
    {
        std::set<uint64_t> seen;
        for (int r = 0; r < img_side; ++r)
        {
            for (int c = 0; c < img_side; ++c)
            {
                auto patch = extract_patch(X[i], img_side, img_side, r, c, ps);
                auto it = lookup.find(patch);
                if (it != lookup.end())
                    seen.insert(it->second);
            }
        }
        for (uint64_t vi : seen)
            D[i][vi] = 1.0f;
    }
    return D;
}

// ============================================================
//  CountingBagOfWords
// ============================================================

CountingBagOfWords::CountingBagOfWords(const ivc::ByteDataset& X,
                                       const size_t patch_size)
    : BagOfWords(X, patch_size)
{}

ivc::FloatDataset CountingBagOfWords::transform(const ivc::ByteDataset& X) const
{
    const size_t V = _vocab.size();
    ivc::FloatDataset D(X.size(), ivc::FloatRow(V, 0.0f));

    if (X.empty() || V == 0) return D;

    std::map<ivc::GrayscaleByteImg, uint64_t, GrayscaleByteImg_comp_t> lookup;
    uint64_t idx = 0;
    for (const auto& p : _vocab.ordered_elements())
        lookup[p] = idx++;

    const int img_pixels = static_cast<int>(X[0].size());
    const int img_side   = static_cast<int>(std::round(std::sqrt(static_cast<double>(img_pixels))));
    const int ps         = static_cast<int>(_patch_size);

    for (size_t i = 0; i < X.size(); ++i)
    {
        for (int r = 0; r < img_side; ++r)
        {
            for (int c = 0; c < img_side; ++c)
            {
                auto patch = extract_patch(X[i], img_side, img_side, r, c, ps);
                auto it = lookup.find(patch);
                if (it != lookup.end())
                    D[i][it->second] += 1.0f;
            }
        }
    }
    return D;
}

// ============================================================
//  OVR (One-vs-Rest)
// ============================================================

<<<<<<< HEAD
struct OVRState
{
    std::vector<ivc::LogReg> classifiers;
    std::vector<float_t>     classes;
    size_t                   num_features = 0;
};

static std::map<const OVR*, OVRState> s_ovr_states;

=======
>>>>>>> parent of 4dfd23f (vbow updated)
OVR::OVR()
    : _classifiers(), _classes()
{}

void OVR::train(const ivc::FloatDataset& X,
                const ivc::ProbVector& y_gt,
                const float_t lr,
                const size_t max_epochs)
{
    if (X.empty()) return;

<<<<<<< HEAD
    OVRState& st = s_ovr_states[this];
=======
    // Collect unique class labels
    std::set<float_t> class_set(y_gt.begin(), y_gt.end());
    _classes.assign(class_set.begin(), class_set.end());
>>>>>>> parent of 4dfd23f (vbow updated)

    _classifiers.clear();

<<<<<<< HEAD
    std::vector<float_t> new_classes(class_set.begin(), class_set.end());
    const size_t nf = static_cast<size_t>(X.cols());

    // reset if class set or feature dimensionality changed
    bool reset = (new_classes != st.classes) ||
                 (st.classifiers.size() != new_classes.size()) ||
                 (st.num_features != nf);

    if (reset)
    {
        st.classes      = new_classes;
        st.num_features = nf;
        st.classifiers.clear();
        for (float_t cls : st.classes)
        {
            ivc::ProbVector binary_y(y_gt.size());
            for (Eigen::Index i = 0; i < y_gt.size(); ++i)
                binary_y(i) = (y_gt(i) == cls) ? 1.0f : 0.0f;
            ivc::LogReg clf(nf);
            clf.train(X, binary_y, lr, max_epochs);
            st.classifiers.push_back(std::move(clf));
        }
    }
    else
    {
        // warm-start: continue from current weights rather than reinitializing
        for (size_t k = 0; k < st.classes.size(); ++k)
        {
            ivc::ProbVector binary_y(y_gt.size());
            for (Eigen::Index i = 0; i < y_gt.size(); ++i)
                binary_y(i) = (y_gt(i) == st.classes[k]) ? 1.0f : 0.0f;
            st.classifiers[k].train(X, binary_y, lr, max_epochs);
        }
=======
    for (float_t cls : _classes)
    {
        // Build binary labels: 1 if sample == cls, 0 otherwise
        ivc::ProbVector binary_y(y_gt.size());
        for (size_t i = 0; i < y_gt.size(); ++i)
            binary_y[i] = (y_gt[i] == cls) ? 1.0f : 0.0f;

        ivc::LogReg clf;
        clf.train(X, binary_y, lr, max_epochs);
        _classifiers.push_back(std::move(clf));
>>>>>>> parent of 4dfd23f (vbow updated)
    }
}

ivc::ProbVector OVR::predict(const ivc::FloatDataset& X) const
{
    const size_t N = X.size();
    ivc::ProbVector y_pred(N, 0.0f);

    if (_classifiers.empty() || _classes.empty()) return y_pred;

    const size_t K = _classes.size();

    // For each sample, pick the class with the highest confidence score
    for (size_t i = 0; i < N; ++i)
    {
        ivc::FloatDataset single_sample = {X[i]};
        float_t best_score = -std::numeric_limits<float_t>::infinity();
<<<<<<< HEAD
        float_t best_class = st.classes[0];
=======
        float_t best_class = _classes[0];

>>>>>>> parent of 4dfd23f (vbow updated)
        for (size_t k = 0; k < K; ++k)
        {
            // predict_proba returns probability of class=1
            ivc::ProbVector proba = _classifiers[k].predict_proba(single_sample);
            float_t score = proba[0]; // probability for this class
            if (score > best_score)
            {
                best_score = score;
                best_class = _classes[k];
            }
        }
        y_pred[i] = best_class;
    }
    return y_pred;
}

// summed cross-entropy from each binary classifier — monotonically non-increasing
float_t OVR::cost(const ivc::FloatDataset& X,
                  const ivc::ProbVector& y_gt) const
{
<<<<<<< HEAD
    auto it = s_ovr_states.find(this);
    if (it == s_ovr_states.end()) return 0.0f;
    const OVRState& st = it->second;
    if (st.classifiers.empty()) return 0.0f;

    float_t total = 0.0f;
    for (size_t k = 0; k < st.classes.size(); ++k)
    {
        ivc::ProbVector binary_y(y_gt.size());
        for (Eigen::Index i = 0; i < y_gt.size(); ++i)
            binary_y(i) = (y_gt(i) == st.classes[k]) ? 1.0f : 0.0f;
        total += st.classifiers[k].cost(X, binary_y);
    }
    return total;
=======
    ivc::ProbVector y_pred = predict(X);
    size_t correct = 0;
    for (size_t i = 0; i < y_gt.size(); ++i)
        if (y_pred[i] == y_gt[i]) ++correct;
    // Return 1 - accuracy as cost
    return 1.0f - static_cast<float_t>(correct) / static_cast<float_t>(y_gt.size());
>>>>>>> parent of 4dfd23f (vbow updated)
}

// ============================================================
//  tile_dataset
// ============================================================

ivc::ByteDataset tile_dataset(const ivc::ByteDataset& X,
                               const size_t level)
{
    if (level == 0 || X.empty()) return X;

    const int img_pixels = static_cast<int>(X[0].size());
    const int img_side   = static_cast<int>(std::round(std::sqrt(static_cast<double>(img_pixels))));

    // Number of tiles per side at this level: 2^level
    const int tiles_per_side = 1 << static_cast<int>(level); // 2 for level=1, 4 for level=2
    const int tile_side = img_side / tiles_per_side;

    ivc::ByteDataset result;
    result.reserve(X.size() * tiles_per_side * tiles_per_side);

    for (const auto& img : X)
    {
        for (int tr = 0; tr < tiles_per_side; ++tr)
        {
            for (int tc = 0; tc < tiles_per_side; ++tc)
            {
                ivc::GrayscaleByteImg tile(tile_side * tile_side);
                for (int r = 0; r < tile_side; ++r)
                {
                    for (int c = 0; c < tile_side; ++c)
                    {
                        int src_r = tr * tile_side + r;
                        int src_c = tc * tile_side + c;
                        tile[r * tile_side + c] = img[src_r * img_side + src_c];
                    }
                }
                result.push_back(std::move(tile));
            }
        }
    }
    return result;
}

// ============================================================
//  HistogramOfGradients
<<<<<<< HEAD
//
//  ivc::sobel_angle(img) → Eigen::VectorXf of angles in degrees
//  one entry per pixel, row-major.
=======
>>>>>>> parent of 4dfd23f (vbow updated)
// ============================================================

HistogramOfGradients::HistogramOfGradients()
{}

ivc::FloatDataset HistogramOfGradients::transform(const ivc::ByteDataset& X) const
{
    // 8 bins of 45 degrees each (0-44, 45-89, ..., 315-359)
    const int NUM_BINS = 8;
    const float_t BIN_WIDTH = 45.0f;

    ivc::FloatDataset D;
    D.reserve(X.size());

    for (const auto& img : X)
    {
        const int img_pixels = static_cast<int>(img.size());
        const int img_side   = static_cast<int>(std::round(std::sqrt(static_cast<double>(img_pixels))));

        // Use provided Sobel angle operator: returns angles in degrees [0, 360)
        // ivc::sobel_angle(img, rows, cols) -> std::vector<float_t>
        std::vector<float_t> angles = ivc::sobel_angle(img, img_side, img_side);

        ivc::FloatRow hist(NUM_BINS, 0.0f);
        for (float_t angle : angles)
        {
            // Normalize angle to [0, 360)
            while (angle < 0.0f)   angle += 360.0f;
            while (angle >= 360.0f) angle -= 360.0f;

            int bin = static_cast<int>(angle / BIN_WIDTH) % NUM_BINS;
            hist[bin] += 1.0f;
        }

        // L1 normalize
        float_t total = 0.0f;
        for (float_t v : hist) total += v;
        if (total > 0.0f)
            for (float_t& v : hist) v /= total;

        D.push_back(std::move(hist));
    }
    return D;
}

// ============================================================
//  balance
// ============================================================

static float_t euclidean_dist(const ivc::FloatRow& a, const ivc::FloatRow& b)
{
    float_t dist = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
    {
        float_t d = a[i] - b[i];
        dist += d * d;
    }
    return std::sqrt(dist);
}

std::tuple<ivc::ByteDataset,
           ivc::ProbVector> balance(const ivc::ByteDataset& X,
                                    const ivc::ProbVector& y_gt,
                                    const ivc::student::balance_type_t balance_type)
{
    if (X.empty()) return std::make_tuple(X, y_gt);

    // Group indices by class label
    std::map<float_t, std::vector<size_t>> class_indices;
    for (size_t i = 0; i < y_gt.size(); ++i)
        class_indices[y_gt[i]].push_back(i);

    // Find majority count
    size_t max_count = 0, min_count = std::numeric_limits<size_t>::max();
    for (const auto& kv : class_indices)
    {
        max_count = std::max(max_count, kv.second.size());
        min_count = std::min(min_count, kv.second.size());
    }

    ivc::ByteDataset X_bal;
    ivc::ProbVector  y_bal;

    std::mt19937 rng(42);

    if (balance_type == OVERSAMPLE)
    {
        for (const auto& kv : class_indices)
        {
            const auto& idxs = kv.second;
            size_t n = idxs.size();
            // Add all existing samples
            for (size_t i : idxs)
            {
                X_bal.push_back(X[i]);
                y_bal.push_back(kv.first);
            }
            // Duplicate until we reach max_count
            std::uniform_int_distribution<size_t> dist(0, n - 1);
            for (size_t added = n; added < max_count; ++added)
            {
                size_t pick = idxs[dist(rng)];
                X_bal.push_back(X[pick]);
                y_bal.push_back(kv.first);
            }
        }
    }
    else if (balance_type == UNDERSAMPLE)
    {
        for (const auto& kv : class_indices)
        {
            std::vector<size_t> idxs = kv.second;
            std::shuffle(idxs.begin(), idxs.end(), rng);
            for (size_t i = 0; i < min_count; ++i)
            {
                X_bal.push_back(X[idxs[i]]);
                y_bal.push_back(kv.first);
            }
        }
    }
    else if (balance_type == SMOTE)
    {
        // SMOTE: operate in image pixel space (feature space)
        const int K_NEIGHBORS = 5;

        for (const auto& kv : class_indices)
        {
            const auto& idxs = kv.second;
            size_t n = idxs.size();
            float_t label = kv.first;

<<<<<<< HEAD
            // Add originals
            for (size_t i : idxs) { X_bal.push_back(*Xv[i]); y_vec.push_back(lbl); }
            if (n >= max_count) continue;

            // if there's only one sample in this class we can't interpolate,
            // so fall back to plain duplication
            if (n == 1)
            {
                size_t to_dup = max_count - 1;
                for (size_t s = 0; s < to_dup; ++s)
                { X_bal.push_back(*Xv[idxs[0]]); y_vec.push_back(lbl); }
                continue;
            }

            const size_t to_gen = max_count - n;

            const ivc::GrayscaleByteImg& ref = *Xv[idxs[0]];
            const int ir = static_cast<int>(ref.rows());
            const int ic = static_cast<int>(ref.cols());

            std::uniform_int_distribution<size_t>   sd(0, n - 1);
            std::uniform_real_distribution<float_t> ad(0.0f, 1.0f);

            for (size_t s = 0; s < to_gen; ++s)
=======
            // Add all original samples
            for (size_t i : idxs)
>>>>>>> parent of 4dfd23f (vbow updated)
            {
                X_bal.push_back(X[i]);
                y_bal.push_back(label);
            }

            if (n >= max_count) continue; // already majority

            size_t to_generate = max_count - n;
            std::uniform_int_distribution<size_t> sample_dist(0, n - 1);

            const int pixel_dim = static_cast<int>(X[0].size());

            for (size_t s = 0; s < to_generate; ++s)
            {
                // Pick random minority sample
                size_t anchor_local = sample_dist(rng);
                size_t anchor_idx   = idxs[anchor_local];

                // Compute distances to all other minority samples
                std::vector<std::pair<float_t, size_t>> dists;
                dists.reserve(n);
                for (size_t j = 0; j < n; ++j)
                {
                    if (j == anchor_local) continue;
                    float_t d = 0.0f;
                    for (int p = 0; p < pixel_dim; ++p)
                    {
                        float_t diff = static_cast<float_t>(X[anchor_idx][p])
                                     - static_cast<float_t>(X[idxs[j]][p]);
                        d += diff * diff;
                    }
                    dists.push_back({std::sqrt(d), j});
                }

                // clamp k to however many neighbors actually exist
                int actual_k = std::min(K_NEIGHBORS, static_cast<int>(dists.size()));
                std::partial_sort(dists.begin(), dists.begin() + actual_k, dists.end());

                // Pick random neighbor among K nearest
                std::uniform_int_distribution<int> neighbor_dist(0, actual_k - 1);
                size_t neighbor_local = dists[neighbor_dist(rng)].second;
                size_t neighbor_idx   = idxs[neighbor_local];

                // Interpolate
                std::uniform_real_distribution<float_t> alpha_dist(0.0f, 1.0f);
                float_t alpha = alpha_dist(rng);

                ivc::GrayscaleByteImg synthetic(pixel_dim);
                for (int p = 0; p < pixel_dim; ++p)
                {
                    float_t v = (1.0f - alpha) * static_cast<float_t>(X[anchor_idx][p])
                               + alpha * static_cast<float_t>(X[neighbor_idx][p]);
                    synthetic[p] = static_cast<uint8_t>(std::round(v));
                }

                X_bal.push_back(std::move(synthetic));
                y_bal.push_back(label);
            }
        }
    }

    return std::make_tuple(std::move(X_bal), std::move(y_bal));
}

} // namespace student
} // namespace ivc
