// SYSTEM INCLUDES
#include <algorithm>
#include <cmath>
#include <list>
#include <map>
#include <random>
#include <set>
#include <tuple>
#include <vector>

#include <585/common/types.h>
#include <585/grad/grad.h>   // sobel_angle (see <585/grad/grad.h> for exact namespace)
#include <585/vbow/vbow.h>   // ivc::LogReg

// C++ PROJECT INCLUDES
#include "vbow/vbow.h"

// ============================================================
//  Confirmed types (from compiler errors):
//
//  GrayscaleByteImg  = Eigen::Matrix<uint8_t,-1,-1>
//    element access: img(r,c)   construction: GrayscaleByteImg(rows,cols)
//
//  ByteDataset       = std::list<GrayscaleByteImg>
//    no operator[]  no reserve()   iterate with range-for
//
//  FloatDataset      = Eigen::Matrix<float,-1,-1>   rows=samples cols=feats
//    element access: D(i,j)   row: D.row(i)   no push_back
//
//  ProbVector        = Eigen::VectorXf
//    element access: v(i)   no push_back
//    construct: ProbVector::Zero(n) or ProbVector(n)
//
//  LogReg            = LogReg(size_t num_features)
//
//  OVR private:      EMPTY in header → external static map for state
// ============================================================

namespace ivc
{
namespace student
{

// ============================================================
//  Internal helpers
// ============================================================

// Build a random-access vector of pointers into a ByteDataset list
static std::vector<const ivc::GrayscaleByteImg*>
make_ptr_index(const ivc::ByteDataset& X)
{
    std::vector<const ivc::GrayscaleByteImg*> idx;
    idx.reserve(X.size());
    for (const auto& img : X)
        idx.push_back(&img);
    return idx;
}

// Extract a ps×ps patch centred at (cr,cc) from a 2-D Eigen image,
// zero-padded outside bounds.
static ivc::GrayscaleByteImg extract_patch(const ivc::GrayscaleByteImg& img,
                                            int cr, int cc, int ps)
{
    const int rows = static_cast<int>(img.rows());
    const int cols = static_cast<int>(img.cols());
    const int half = ps / 2;

    ivc::GrayscaleByteImg patch(ps, ps);
    patch.setZero();

    for (int dr = 0; dr < ps; ++dr)
        for (int dc = 0; dc < ps; ++dc)
        {
            int r = cr - half + dr;
            int c = cc - half + dc;
            if (r >= 0 && r < rows && c >= 0 && c < cols)
                patch(dr, dc) = img(r, c);
        }
    return patch;
}

// ============================================================
//  Vocab
// ============================================================

Vocab::Vocab()
    : _patch_to_idx(), _idx_to_patch()
{}

void Vocab::add(const ivc::GrayscaleByteImg& e)
{
    if (_patch_to_idx.find(e) == _patch_to_idx.end())
    {
        uint64_t idx = static_cast<uint64_t>(_patch_to_idx.size());
        _patch_to_idx[e]   = idx;
        _idx_to_patch[idx] = e;
    }
}

const size_t Vocab::size() const
{
    return _patch_to_idx.size();
}

std::list<ivc::GrayscaleByteImg> Vocab::ordered_elements() const
{
    std::list<ivc::GrayscaleByteImg> elements;
    for (const auto& kv : _idx_to_patch)   // std::map → ascending key order
        elements.push_back(kv.second);
    return elements;
}

// ============================================================
//  BagOfWords base — builds vocabulary from every pixel patch
// ============================================================

BagOfWords::BagOfWords(const ivc::ByteDataset& X, const size_t patch_size)
    : _vocab(), _patch_size(patch_size)
{
    const int ps = static_cast<int>(patch_size);
    for (const auto& img : X)
    {
        const int rows = static_cast<int>(img.rows());
        const int cols = static_cast<int>(img.cols());
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                _vocab.add(extract_patch(img, r, c, ps));
    }
}

// ============================================================
//  Shared BoW transform kernel
//  accumulate=false → binary (set 1 on first hit)
//  accumulate=true  → counting (increment)
// ============================================================

static ivc::FloatDataset bow_transform_impl(const ivc::ByteDataset& X,
                                             const Vocab& vocab,
                                             int ps,
                                             bool accumulate)
{
    const Eigen::Index V = static_cast<Eigen::Index>(vocab.size());
    const Eigen::Index N = static_cast<Eigen::Index>(X.size());

    ivc::FloatDataset D = ivc::FloatDataset::Zero(N, V);
    if (N == 0 || V == 0) return D;

    // patch → column index
    std::map<ivc::GrayscaleByteImg, Eigen::Index, GrayscaleByteImg_comp_t> lookup;
    {
        Eigen::Index col = 0;
        for (const auto& p : vocab.ordered_elements())
            lookup[p] = col++;
    }

    Eigen::Index row = 0;
    for (const auto& img : X)
    {
        const int rows = static_cast<int>(img.rows());
        const int cols = static_cast<int>(img.cols());
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
            {
                auto patch = extract_patch(img, r, c, ps);
                auto it = lookup.find(patch);
                if (it != lookup.end())
                {
                    if (accumulate)
                        D(row, it->second) += 1.0f;
                    else
                        D(row, it->second) = 1.0f;
                }
            }
        ++row;
    }
    return D;
}

// ============================================================
//  BinaryBagOfWords
// ============================================================

BinaryBagOfWords::BinaryBagOfWords(const ivc::ByteDataset& X, const size_t patch_size)
    : BagOfWords(X, patch_size)
{}

ivc::FloatDataset BinaryBagOfWords::transform(const ivc::ByteDataset& X) const
{
    return bow_transform_impl(X, _vocab, static_cast<int>(_patch_size), false);
}

// ============================================================
//  CountingBagOfWords
// ============================================================

CountingBagOfWords::CountingBagOfWords(const ivc::ByteDataset& X, const size_t patch_size)
    : BagOfWords(X, patch_size)
{}

ivc::FloatDataset CountingBagOfWords::transform(const ivc::ByteDataset& X) const
{
    return bow_transform_impl(X, _vocab, static_cast<int>(_patch_size), true);
}

// ============================================================
//  OVR — One-vs-Rest
//
//  OVR's private section in the header is empty, so we store
//  per-instance state in a file-static map keyed on `this`.
// ============================================================

struct OVRState
{
    std::vector<ivc::LogReg> classifiers;
    std::vector<float_t>     classes;
};

static std::map<const OVR*, OVRState> s_ovr_states;

OVR::OVR()
{
    s_ovr_states.emplace(this, OVRState{});
}

void OVR::train(const ivc::FloatDataset& X,
                const ivc::ProbVector&   y_gt,
                const float_t            lr,
                const size_t             max_epochs)
{
    if (X.rows() == 0) return;

    OVRState& st = s_ovr_states[this];
    st.classifiers.clear();
    st.classes.clear();

    std::set<float_t> class_set;
    for (Eigen::Index i = 0; i < y_gt.size(); ++i)
        class_set.insert(y_gt(i));

    st.classes.assign(class_set.begin(), class_set.end());

    const size_t nf = static_cast<size_t>(X.cols());

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

ivc::ProbVector OVR::predict(const ivc::FloatDataset& X) const
{
    const Eigen::Index N = X.rows();
    ivc::ProbVector y_pred = ivc::ProbVector::Zero(N);

    auto it = s_ovr_states.find(this);
    if (it == s_ovr_states.end()) return y_pred;
    const OVRState& st = it->second;
    if (st.classifiers.empty()) return y_pred;

    const size_t K = st.classes.size();

    for (Eigen::Index i = 0; i < N; ++i)
    {
        // Copy single row into a 1×D FloatDataset for predict_proba
        ivc::FloatDataset single = X.row(i);

        float_t best_score = -std::numeric_limits<float_t>::infinity();
        float_t best_class = st.classes[0];

        for (size_t k = 0; k < K; ++k)
        {
            // LogReg::predict() returns a ProbVector of sigmoid scores (one per sample).
            // For a 1-row input this gives a length-1 vector; score(0) is P(y=1|x).
            ivc::ProbVector proba = st.classifiers[k].predict(single);
            float_t score = proba(0);
            if (score > best_score) { best_score = score; best_class = st.classes[k]; }
        }
        y_pred(i) = best_class;
    }
    return y_pred;
}

float_t OVR::cost(const ivc::FloatDataset& X,
                  const ivc::ProbVector&   y_gt) const
{
    ivc::ProbVector y_pred = predict(X);
    Eigen::Index correct = 0;
    for (Eigen::Index i = 0; i < y_gt.size(); ++i)
        if (y_pred(i) == y_gt(i)) ++correct;
    return 1.0f - static_cast<float_t>(correct) /
                  static_cast<float_t>(y_gt.size());
}

// ============================================================
//  tile_dataset
// ============================================================

ivc::ByteDataset tile_dataset(const ivc::ByteDataset& X, const size_t level)
{
    if (level == 0 || X.empty()) return X;

    const int tps = 1 << static_cast<int>(level);  // tiles per side

    ivc::ByteDataset result;
    for (const auto& img : X)
    {
        const int rows      = static_cast<int>(img.rows());
        const int cols      = static_cast<int>(img.cols());
        const int tile_rows = rows / tps;
        const int tile_cols = cols / tps;

        for (int tr = 0; tr < tps; ++tr)
            for (int tc = 0; tc < tps; ++tc)
            {
                // Eigen .block() returns a sub-matrix expression; assign to
                // a new GrayscaleByteImg to force a copy.
                ivc::GrayscaleByteImg tile =
                    img.block(tr * tile_rows, tc * tile_cols,
                              tile_rows, tile_cols);
                result.push_back(std::move(tile));
            }
    }
    return result;
}

// ============================================================
//  HistogramOfGradients
//
//  Uses ivc::get_sobel_3x3_gradient_angles(img) from <585/grad/grad.h>
//  Returns ivc::GrayscaleFloatImg = Eigen::Matrix<float,-1,-1>
//  with one angle per pixel (degrees). Handles padding internally.
// ============================================================

HistogramOfGradients::HistogramOfGradients()
{}

ivc::FloatDataset HistogramOfGradients::transform(const ivc::ByteDataset& X) const
{
    const int          NUM_BINS  = 8;
    const float_t      BIN_WIDTH = 45.0f;
    const Eigen::Index N         = static_cast<Eigen::Index>(X.size());

    ivc::FloatDataset D = ivc::FloatDataset::Zero(N, NUM_BINS);
    if (N == 0) return D;

    Eigen::Index row = 0;
    for (const auto& img : X)
    {
        // ivc::get_sobel_3x3_gradient_angles returns a GrayscaleFloatImg
        // (Eigen::Matrix<float,-1,-1>) with per-pixel angles in degrees.
        // It pads internally so we pass img directly.
        ivc::GrayscaleFloatImg angle_img = ivc::get_sobel_3x3_gradient_angles(img);

        Eigen::VectorXf hist = Eigen::VectorXf::Zero(NUM_BINS);
        for (int r = 0; r < static_cast<int>(angle_img.rows()); ++r)
            for (int c = 0; c < static_cast<int>(angle_img.cols()); ++c)
            {
                float_t a = std::fmod(angle_img(r, c), 360.0f);
                if (a < 0.0f) a += 360.0f;
                int bin = static_cast<int>(a / BIN_WIDTH) % NUM_BINS;
                hist(bin) += 1.0f;
            }

        float_t total = hist.sum();
        if (total > 0.0f) hist /= total;

        D.row(row) = hist.transpose();
        ++row;
    }
    return D;
}

// ============================================================
//  balance
// ============================================================

std::tuple<ivc::ByteDataset, ivc::ProbVector>
balance(const ivc::ByteDataset&            X,
        const ivc::ProbVector&             y_gt,
        const ivc::student::balance_type_t balance_type)
{
    if (X.empty())
        return std::make_tuple(X, y_gt);

    // Random-access into list
    std::vector<const ivc::GrayscaleByteImg*> Xv = make_ptr_index(X);
    const size_t N = Xv.size();

    // Group indices by class
    std::map<float_t, std::vector<size_t>> class_idx;
    for (size_t i = 0; i < N; ++i)
        class_idx[y_gt(static_cast<Eigen::Index>(i))].push_back(i);

    size_t max_count = 0, min_count = std::numeric_limits<size_t>::max();
    for (const auto& kv : class_idx)
    {
        max_count = std::max(max_count, kv.second.size());
        min_count = std::min(min_count, kv.second.size());
    }

    ivc::ByteDataset     X_bal;
    std::vector<float_t> y_vec;

    std::mt19937 rng(42);

    if (balance_type == OVERSAMPLE)
    {
        for (const auto& kv : class_idx)
        {
            const auto& idxs = kv.second;
            for (size_t i : idxs) { X_bal.push_back(*Xv[i]); y_vec.push_back(kv.first); }
            size_t added = idxs.size();
            std::uniform_int_distribution<size_t> dist(0, idxs.size() - 1);
            while (added < max_count)
            {
                size_t pick = idxs[dist(rng)];
                X_bal.push_back(*Xv[pick]);
                y_vec.push_back(kv.first);
                ++added;
            }
        }
    }
    else if (balance_type == UNDERSAMPLE)
    {
        for (const auto& kv : class_idx)
        {
            std::vector<size_t> idxs = kv.second;
            std::shuffle(idxs.begin(), idxs.end(), rng);
            for (size_t i = 0; i < min_count; ++i)
            { X_bal.push_back(*Xv[idxs[i]]); y_vec.push_back(kv.first); }
        }
    }
    else if (balance_type == SMOTE)
    {
        const int K_NEIGHBORS = 5;

        for (const auto& kv : class_idx)
        {
            const auto& idxs  = kv.second;
            const float_t lbl = kv.first;
            const size_t  n   = idxs.size();

            // Add originals
            for (size_t i : idxs) { X_bal.push_back(*Xv[i]); y_vec.push_back(lbl); }
            if (n >= max_count) continue;

            const size_t to_gen = max_count - n;

            const ivc::GrayscaleByteImg& ref = *Xv[idxs[0]];
            const int ir = static_cast<int>(ref.rows());
            const int ic = static_cast<int>(ref.cols());

            std::uniform_int_distribution<size_t>   sd(0, n - 1);
            std::uniform_real_distribution<float_t> ad(0.0f, 1.0f);

            for (size_t s = 0; s < to_gen; ++s)
            {
                size_t al = sd(rng);
                size_t ai = idxs[al];
                const ivc::GrayscaleByteImg& aimg = *Xv[ai];

                // Distances to other minority samples
                std::vector<std::pair<float_t, size_t>> dists;
                dists.reserve(n - 1);
                for (size_t j = 0; j < n; ++j)
                {
                    if (j == al) continue;
                    const ivc::GrayscaleByteImg& nimg = *Xv[idxs[j]];
                    float_t d = 0.0f;
                    for (int rr = 0; rr < ir; ++rr)
                        for (int cc = 0; cc < ic; ++cc)
                        {
                            float_t diff = static_cast<float_t>(aimg(rr, cc))
                                         - static_cast<float_t>(nimg(rr, cc));
                            d += diff * diff;
                        }
                    dists.push_back({std::sqrt(d), j});
                }

                int actual_k = std::min(K_NEIGHBORS, static_cast<int>(dists.size()));
                std::partial_sort(dists.begin(), dists.begin() + actual_k, dists.end());

                std::uniform_int_distribution<int> nd(0, actual_k - 1);
                size_t nl = dists[nd(rng)].second;
                const ivc::GrayscaleByteImg& nimg = *Xv[idxs[nl]];

                float_t alpha = ad(rng);
                ivc::GrayscaleByteImg syn(ir, ic);
                for (int rr = 0; rr < ir; ++rr)
                    for (int cc = 0; cc < ic; ++cc)
                    {
                        float_t v = (1.0f - alpha) * static_cast<float_t>(aimg(rr, cc))
                                  + alpha           * static_cast<float_t>(nimg(rr, cc));
                        syn(rr, cc) = static_cast<uint8_t>(std::round(v));
                    }

                X_bal.push_back(std::move(syn));
                y_vec.push_back(lbl);
            }
        }
    }

    // Pack y_vec → VectorXf
    ivc::ProbVector y_bal(static_cast<Eigen::Index>(y_vec.size()));
    for (Eigen::Index i = 0; i < y_bal.size(); ++i)
        y_bal(i) = y_vec[static_cast<size_t>(i)];

    return std::make_tuple(std::move(X_bal), std::move(y_bal));
}

} // end of namespace student
} // end of namespace ivc
