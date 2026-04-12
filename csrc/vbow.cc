#include <algorithm>
#include <cmath>
#include <list>
#include <map>
#include <random>
#include <set>
#include <tuple>
#include <vector>

#include <585/common/types.h>
#include <585/grad/grad.h>
#include <585/vbow/vbow.h>

#include "vbow/vbow.h"

namespace ivc
{
namespace student
{

// ByteDataset is a std::list so we need a pointer index for O(1) random access
static std::vector<const ivc::GrayscaleByteImg*>
make_ptr_index(const ivc::ByteDataset& X)
{
    std::vector<const ivc::GrayscaleByteImg*> idx;
    idx.reserve(X.size());
    for (const auto& img : X)
        idx.push_back(&img);
    return idx;
}

// extract a ps x ps patch centered at (cr, cc), zero-padding outside borders
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

// _idx_to_patch is std::map<uint64_t,...> so iteration is already in index order
std::list<ivc::GrayscaleByteImg> Vocab::ordered_elements() const
{
    std::list<ivc::GrayscaleByteImg> elements;
    for (const auto& kv : _idx_to_patch)
        elements.push_back(kv.second);
    return elements;
}

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

// build the patch->column index lookup once rather than on every transform() call
static std::map<ivc::GrayscaleByteImg, Eigen::Index, GrayscaleByteImg_comp_t>
build_lookup(const Vocab& vocab)
{
    std::map<ivc::GrayscaleByteImg, Eigen::Index, GrayscaleByteImg_comp_t> lookup;
    Eigen::Index col = 0;
    for (const auto& p : vocab.ordered_elements())
        lookup[p] = col++;
    return lookup;
}

static ivc::FloatDataset bow_transform_impl(const ivc::ByteDataset& X,
                                             const Vocab& vocab,
                                             int ps,
                                             bool accumulate)
{
    const Eigen::Index V = static_cast<Eigen::Index>(vocab.size());
    const Eigen::Index N = static_cast<Eigen::Index>(X.size());

    ivc::FloatDataset D = ivc::FloatDataset::Zero(N, V);
    if (N == 0 || V == 0) return D;

    const auto lookup = build_lookup(vocab);

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

BinaryBagOfWords::BinaryBagOfWords(const ivc::ByteDataset& X, const size_t patch_size)
    : BagOfWords(X, patch_size)
{}

ivc::FloatDataset BinaryBagOfWords::transform(const ivc::ByteDataset& X) const
{
    return bow_transform_impl(X, _vocab, static_cast<int>(_patch_size), false);
}

CountingBagOfWords::CountingBagOfWords(const ivc::ByteDataset& X, const size_t patch_size)
    : BagOfWords(X, patch_size)
{}

ivc::FloatDataset CountingBagOfWords::transform(const ivc::ByteDataset& X) const
{
    return bow_transform_impl(X, _vocab, static_cast<int>(_patch_size), true);
}

// OVR has no private fields in the header so state lives here keyed on `this`
struct OVRState
{
    std::vector<ivc::LogReg> classifiers;
    std::vector<float_t>     classes;
    size_t                   num_features = 0;
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

    std::set<float_t> class_set;
    for (Eigen::Index i = 0; i < y_gt.size(); ++i)
        class_set.insert(y_gt(i));

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
        // slice one row as a 1xD matrix so predict() returns a length-1 probability
        ivc::FloatDataset single = X.row(i);

        float_t best_score = -std::numeric_limits<float_t>::infinity();
        float_t best_class = st.classes[0];
        for (size_t k = 0; k < K; ++k)
        {
            ivc::ProbVector proba = st.classifiers[k].predict(single);
            float_t s = proba(0);
            if (s > best_score) { best_score = s; best_class = st.classes[k]; }
        }
        y_pred(i) = best_class;
    }
    return y_pred;
}

// use summed cross-entropy from each binary classifier so cost is monotonically
// non-increasing during training (accuracy-based cost would not be)
float_t OVR::cost(const ivc::FloatDataset& X,
                  const ivc::ProbVector&   y_gt) const
{
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
}

ivc::ByteDataset tile_dataset(const ivc::ByteDataset& X, const size_t level)
{
    if (level == 0 || X.empty()) return X;

    const int tps = 1 << static_cast<int>(level);

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
                ivc::GrayscaleByteImg tile =
                    img.block(tr * tile_rows, tc * tile_cols, tile_rows, tile_cols);
                result.push_back(std::move(tile));
            }
    }
    return result;
}

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
        // handles padding internally, returns per-pixel angles in degrees
        ivc::GrayscaleFloatImg angle_img = ivc::get_sobel_3x3_gradient_angles(img);

        Eigen::VectorXf hist = Eigen::VectorXf::Zero(NUM_BINS);
        for (int r = 0; r < static_cast<int>(angle_img.rows()); ++r)
            for (int c = 0; c < static_cast<int>(angle_img.cols()); ++c)
            {
                // get_sobel_3x3_gradient_angles uses a different kernel convention
                // than standard atan2(Gy,Gx), requiring a 292.5 degree offset
                // to align its output with the expected 45-degree bins
                float_t a = std::fmod(angle_img(r, c) + 292.5f, 360.0f);
                if (a < 0.0f) a += 360.0f;
                int bin = static_cast<int>(a / BIN_WIDTH) % NUM_BINS;
                hist(bin) += 1.0f;
            }

        D.row(row) = hist.transpose();
        ++row;
    }
    return D;
}

std::tuple<ivc::ByteDataset, ivc::ProbVector>
balance(const ivc::ByteDataset&            X,
        const ivc::ProbVector&             y_gt,
        const ivc::student::balance_type_t balance_type)
{
    if (X.empty())
        return std::make_tuple(X, y_gt);

    std::vector<const ivc::GrayscaleByteImg*> Xv = make_ptr_index(X);
    const size_t N = Xv.size();

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
                X_bal.push_back(*Xv[idxs[dist(rng)]]);
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
            {
                size_t al = sd(rng);
                const ivc::GrayscaleByteImg& aimg = *Xv[idxs[al]];

                // compute distances to all other samples in this class
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
                    dists.push_back({d, j}); // store squared dist, avoids sqrt
                }

                // clamp k to however many neighbors actually exist
                int actual_k = std::min(K_NEIGHBORS, static_cast<int>(dists.size()));
                std::partial_sort(dists.begin(), dists.begin() + actual_k, dists.end());

                std::uniform_int_distribution<int> nd(0, actual_k - 1);
                const ivc::GrayscaleByteImg& nimg = *Xv[idxs[dists[nd(rng)].second]];

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

    ivc::ProbVector y_bal(static_cast<Eigen::Index>(y_vec.size()));
    for (Eigen::Index i = 0; i < y_bal.size(); ++i)
        y_bal(i) = y_vec[static_cast<size_t>(i)];

    return std::make_tuple(std::move(X_bal), std::move(y_bal));
}

} // namespace student
} // namespace ivc
