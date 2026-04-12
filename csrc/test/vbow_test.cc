// SYSTEM INCLUDES
#include <cmath>
#include <list>
#include <map>
#include <set>
#include <vector>

#include <gtest/gtest.h>
#include <585/common/types.h>

// C++ PROJECT INCLUDES
#include "vbow/vbow.h"

// ============================================================
//  Helpers
//  GrayscaleByteImg = Eigen::Matrix<uint8_t,-1,-1>
//  ByteDataset      = std::list<GrayscaleByteImg>
//  FloatDataset     = Eigen::Matrix<float,-1,-1>
//  ProbVector       = Eigen::VectorXf
// ============================================================

// Build a square image filled with a constant value
static ivc::GrayscaleByteImg make_img(int side, uint8_t val)
{
    ivc::GrayscaleByteImg img(side, side);
    img.fill(val);
    return img;
}

// Build a dataset from a list of fill values; all images are side×side
static ivc::ByteDataset make_dataset(const std::vector<uint8_t>& vals, int side = 4)
{
    ivc::ByteDataset X;
    for (uint8_t v : vals)
        X.push_back(make_img(side, v));
    return X;
}

// Build a FloatDataset (N×D) from a std::vector<std::vector<float>>
static ivc::FloatDataset make_float_ds(const std::vector<std::vector<float>>& rows)
{
    int N = static_cast<int>(rows.size());
    int D = N > 0 ? static_cast<int>(rows[0].size()) : 0;
    ivc::FloatDataset M(N, D);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j)
            M(i, j) = rows[static_cast<size_t>(i)][static_cast<size_t>(j)];
    return M;
}

// Build a ProbVector from std::vector<float>
static ivc::ProbVector make_prob(const std::vector<float>& v)
{
    ivc::ProbVector p(static_cast<Eigen::Index>(v.size()));
    for (Eigen::Index i = 0; i < p.size(); ++i)
        p(i) = v[static_cast<size_t>(i)];
    return p;
}

// ============================================================
//  Vocab Tests
// ============================================================

TEST(test_vocab, add_unique)
{
    ivc::student::Vocab vocab;
    ivc::GrayscaleByteImg p1 = make_img(2, 10);
    ivc::GrayscaleByteImg p2 = make_img(2, 20);
    vocab.add(p1);
    vocab.add(p2);
    EXPECT_EQ(vocab.size(), 2u);
}

TEST(test_vocab, add_duplicate_no_grow)
{
    ivc::student::Vocab vocab;
    ivc::GrayscaleByteImg p = make_img(2, 42);
    vocab.add(p);
    vocab.add(p);
    vocab.add(p);
    EXPECT_EQ(vocab.size(), 1u);
}

TEST(test_vocab, ordered_elements_insertion_order)
{
    ivc::student::Vocab vocab;
    ivc::GrayscaleByteImg p0 = make_img(2, 10);
    ivc::GrayscaleByteImg p1 = make_img(2, 20);
    ivc::GrayscaleByteImg p2 = make_img(2, 30);

    vocab.add(p0);
    vocab.add(p1);
    vocab.add(p2);

    auto elems = vocab.ordered_elements();
    ASSERT_EQ(elems.size(), 3u);

    auto it = elems.begin();
    EXPECT_TRUE((*it).isApprox(p0)); ++it;
    EXPECT_TRUE((*it).isApprox(p1)); ++it;
    EXPECT_TRUE((*it).isApprox(p2));
}

TEST(test_vocab, ordered_elements_deterministic)
{
    ivc::student::Vocab v1, v2;
    ivc::GrayscaleByteImg pa = make_img(2, 1);
    ivc::GrayscaleByteImg pb = make_img(2, 2);
    ivc::GrayscaleByteImg pc = make_img(2, 3);

    for (auto* v : {&v1, &v2}) { v->add(pa); v->add(pb); v->add(pc); }

    auto e1 = v1.ordered_elements();
    auto e2 = v2.ordered_elements();
    ASSERT_EQ(e1.size(), e2.size());

    auto it1 = e1.begin(), it2 = e2.begin();
    while (it1 != e1.end()) { EXPECT_TRUE(it1->isApprox(*it2)); ++it1; ++it2; }
}

TEST(test_vocab, empty_vocab)
{
    ivc::student::Vocab vocab;
    EXPECT_EQ(vocab.size(), 0u);
    EXPECT_TRUE(vocab.ordered_elements().empty());
}

// ============================================================
//  BinaryBagOfWords Tests
// ============================================================

TEST(test_binary_vbow, output_dimensions)
{
    auto X_train = make_dataset({0, 128, 255}, 4);
    auto X_test  = make_dataset({0, 255},      4);
    ivc::student::BinaryBagOfWords bow(X_train, 2);
    auto D = bow.transform(X_test);

    // 2 test images, V columns
    EXPECT_EQ(D.rows(), 2);
    EXPECT_GT(D.cols(), 0);
}

TEST(test_binary_vbow, values_are_binary)
{
    auto X = make_dataset({0, 64, 128, 192, 255}, 4);
    ivc::student::BinaryBagOfWords bow(X, 2);
    auto D = bow.transform(X);

    for (int i = 0; i < D.rows(); ++i)
        for (int j = 0; j < D.cols(); ++j)
        {
            float v = D(i, j);
            EXPECT_TRUE(v == 0.0f || v == 1.0f) << "Non-binary value: " << v;
        }
}

TEST(test_binary_vbow, train_set_has_nonzero_rows)
{
    auto X = make_dataset({0, 128, 255}, 4);
    ivc::student::BinaryBagOfWords bow(X, 2);
    auto D = bow.transform(X);

    for (int i = 0; i < D.rows(); ++i)
        EXPECT_GT(D.row(i).sum(), 0.0f) << "Row " << i << " is all-zero";
}

TEST(test_binary_vbow, patch_size_1_vocab_equals_unique_pixels)
{
    // patch_size=1: each pixel is its own patch; 2×2 image with values 10,20
    auto X_train = make_dataset({10, 20}, 2);
    ivc::student::BinaryBagOfWords bow(X_train, 1);
    auto D = bow.transform(X_train);
    // 2 unique pixel values → vocab size = 2
    EXPECT_EQ(D.cols(), 2);
    EXPECT_EQ(D.rows(), 2);
}

// ============================================================
//  CountingBagOfWords Tests
// ============================================================

TEST(test_counting_vbow, output_dimensions)
{
    auto X_train = make_dataset({0, 128, 255}, 4);
    auto X_test  = make_dataset({0, 255},      4);
    ivc::student::CountingBagOfWords bow(X_train, 2);
    auto D = bow.transform(X_test);
    EXPECT_EQ(D.rows(), 2);
    EXPECT_GT(D.cols(), 0);
}

TEST(test_counting_vbow, counts_are_nonnegative)
{
    auto X = make_dataset({0, 128, 255}, 4);
    ivc::student::CountingBagOfWords bow(X, 2);
    auto D = bow.transform(X);
    for (int i = 0; i < D.rows(); ++i)
        for (int j = 0; j < D.cols(); ++j)
            EXPECT_GE(D(i, j), 0.0f);
}

TEST(test_counting_vbow, counts_geq_binary)
{
    auto X = make_dataset({42, 84}, 4);
    ivc::student::BinaryBagOfWords  bow_bin(X, 2);
    ivc::student::CountingBagOfWords bow_cnt(X, 2);
    auto Db = bow_bin.transform(X);
    auto Dc = bow_cnt.transform(X);

    ASSERT_EQ(Db.rows(), Dc.rows());
    ASSERT_EQ(Db.cols(), Dc.cols());
    for (int i = 0; i < Db.rows(); ++i)
        for (int j = 0; j < Db.cols(); ++j)
            EXPECT_GE(Dc(i, j), Db(i, j));
}

TEST(test_counting_vbow, row_sum_equals_total_patches)
{
    // 2×2 image, patch_size=1 → 4 patches per image, all same pixel value
    // → one vocab word per unique pixel; count in that word's slot = 4
    auto X_train = make_dataset({50, 100}, 2);
    ivc::student::CountingBagOfWords bow(X_train, 1);
    auto D = bow.transform(X_train);
    for (int i = 0; i < D.rows(); ++i)
        EXPECT_FLOAT_EQ(D.row(i).sum(), 4.0f);
}

// ============================================================
//  OVR Tests
// ============================================================

TEST(test_ovr, separable_1d)
{
<<<<<<< HEAD
    // Class 0: x=0, class 1: x=1
    auto X_train = make_float_ds({{0},{0},{0},{1},{1},{1}});
    auto y_train = make_prob({0,0,0,1,1,1});

    ivc::student::OVR ovr;
    ovr.train(X_train, y_train, 0.1f, 300);

    auto X_test = make_float_ds({{0},{1}});
=======
    // simple linearly separable problem: class 0 at x=0, class 1 at x=10
    // use well-separated values and enough training to ensure convergence
    auto X_train = make_float_ds({{0},{0},{0},{10},{10},{10}});
    auto y_train = make_prob({0,0,0,1,1,1});

    ivc::student::OVR ovr;
    ovr.train(X_train, y_train, 0.01f, 500);

    auto X_test = make_float_ds({{0},{10}});
>>>>>>> 70fae431189c5a359c80acee728e81ef2a122ca8
    auto y_pred = ovr.predict(X_test);

    EXPECT_FLOAT_EQ(y_pred(0), 0.0f);
    EXPECT_FLOAT_EQ(y_pred(1), 1.0f);
}

<<<<<<< HEAD
TEST(test_ovr, cost_in_unit_range)
=======
TEST(test_ovr, cost_is_nonneg_and_finite)
>>>>>>> 70fae431189c5a359c80acee728e81ef2a122ca8
{
    auto X = make_float_ds({{0},{1},{2}});
    auto y = make_prob({0,1,2});
    ivc::student::OVR ovr;
    ovr.train(X, y, 0.01f, 10);
    float c = ovr.cost(X, y);
    EXPECT_GE(c, 0.0f);
    EXPECT_TRUE(std::isfinite(c));
}

TEST(test_ovr, predict_length_matches_input)
{
    auto X_train = make_float_ds({{0},{1}});
    auto y_train = make_prob({0,1});
    ivc::student::OVR ovr;
    ovr.train(X_train, y_train, 0.01f, 10);

    auto X_test = make_float_ds({{0},{1},{0},{1},{0}});
    auto y_pred = ovr.predict(X_test);
    EXPECT_EQ(y_pred.size(), 5);
}

TEST(test_ovr, empty_train_no_crash)
{
    ivc::student::OVR ovr;
    ivc::FloatDataset Xe(0, 0);
    ivc::ProbVector   ye(0);
    EXPECT_NO_THROW(ovr.train(Xe, ye, 0.1f, 10));
    EXPECT_NO_THROW(ovr.predict(Xe));
}

// ============================================================
//  tile_dataset Tests
// ============================================================

TEST(test_tile, level_0_is_identity)
{
    auto X = make_dataset({0, 255}, 4);
    auto T = ivc::student::tile_dataset(X, 0);
    EXPECT_EQ(T.size(), X.size());
    auto itX = X.begin(), itT = T.begin();
    while (itX != X.end()) { EXPECT_TRUE(itX->isApprox(*itT)); ++itX; ++itT; }
}

TEST(test_tile, level_1_quadruples_count)
{
    auto X = make_dataset({128}, 4);   // 1 image
    auto T = ivc::student::tile_dataset(X, 1);
    EXPECT_EQ(T.size(), 4u);            // 4 tiles
    for (const auto& tile : T)
    {
        EXPECT_EQ(tile.rows(), 2);
        EXPECT_EQ(tile.cols(), 2);
    }
}

TEST(test_tile, level_2_produces_16_tiles)
{
    // 8×8 image → level 2 → 16 tiles of 2×2
    ivc::GrayscaleByteImg img(8, 8);
    img.fill(1);
    ivc::ByteDataset X; X.push_back(img);
    auto T = ivc::student::tile_dataset(X, 2);
    EXPECT_EQ(T.size(), 16u);
    for (const auto& tile : T) { EXPECT_EQ(tile.rows(), 2); EXPECT_EQ(tile.cols(), 2); }
}

TEST(test_tile, level_1_pixel_values_preserved)
{
    // Build a 4×4 image where the top-left quadrant is all-0, others differ
    ivc::GrayscaleByteImg img(4, 4);
    img.block(0,0,2,2).fill(0);
    img.block(0,2,2,2).fill(64);
    img.block(2,0,2,2).fill(128);
    img.block(2,2,2,2).fill(255);

    ivc::ByteDataset X; X.push_back(img);
    auto T = ivc::student::tile_dataset(X, 1);
    ASSERT_EQ(T.size(), 4u);

    std::vector<uint8_t> expected_vals = {0, 64, 128, 255};
    size_t idx = 0;
    for (const auto& tile : T)
    {
        // All pixels in each tile should equal the expected fill value
        for (int r = 0; r < tile.rows(); ++r)
            for (int c = 0; c < tile.cols(); ++c)
                EXPECT_EQ(tile(r, c), expected_vals[idx]);
        ++idx;
    }
}

// ============================================================
//  HistogramOfGradients Tests
// ============================================================

TEST(test_hog, output_has_8_bins_per_image)
{
    auto X = make_dataset({128}, 8);
    ivc::student::HistogramOfGradients hog;
    auto D = hog.transform(X);
    EXPECT_EQ(D.rows(), 1);
    EXPECT_EQ(D.cols(), 8);
}

<<<<<<< HEAD
TEST(test_hog, hist_sums_to_one_or_zero)
{
    // Constant image → zero gradient; hist may be all-zero (sum=0) or
    // normalised (sum≈1) depending on how sobel_angle handles flat regions.
=======
TEST(test_hog, hist_sums_to_pixel_count_or_zero)
{
    // HOG outputs raw bin counts (not normalized).
    // A constant image has zero gradient everywhere so all bins should be 0,
    // or if sobel produces some edge at the border the counts sum to <= rows*cols.
>>>>>>> 70fae431189c5a359c80acee728e81ef2a122ca8
    auto X = make_dataset({100}, 8);
    ivc::student::HistogramOfGradients hog;
    auto D = hog.transform(X);
    ASSERT_EQ(D.rows(), 1);
    float s = D.row(0).sum();
<<<<<<< HEAD
    EXPECT_TRUE(std::abs(s - 1.0f) < 1e-4f || std::abs(s) < 1e-6f)
        << "Histogram sum = " << s;
=======
    // sum must be non-negative and <= total pixels (8*8=64)
    EXPECT_GE(s, 0.0f);
    EXPECT_LE(s, 64.0f);
>>>>>>> 70fae431189c5a359c80acee728e81ef2a122ca8
}

TEST(test_hog, all_values_nonnegative)
{
    auto X = make_dataset({50, 100, 200}, 8);
    ivc::student::HistogramOfGradients hog;
    auto D = hog.transform(X);
    for (int i = 0; i < D.rows(); ++i)
        for (int j = 0; j < D.cols(); ++j)
            EXPECT_GE(D(i, j), 0.0f);
}

TEST(test_hog, multiple_images_independent_rows)
{
    auto X = make_dataset({0, 255}, 8);
    ivc::student::HistogramOfGradients hog;
    auto D = hog.transform(X);
    EXPECT_EQ(D.rows(), 2);
    EXPECT_EQ(D.cols(), 8);
}

// ============================================================
//  balance Tests
// ============================================================

// Helper: count occurrences of each label in a ProbVector
static std::map<float, int> count_labels(const ivc::ProbVector& y)
{
    std::map<float, int> counts;
    for (Eigen::Index i = 0; i < y.size(); ++i)
        counts[y(i)]++;
    return counts;
}

TEST(test_balance, oversample_equalizes)
{
    // 3 of class 0, 1 of class 1
    auto X = make_dataset({10, 20, 30, 200}, 4);
    auto y = make_prob({0, 0, 0, 1});

    auto [Xb, yb] = ivc::student::balance(X, y, ivc::student::OVERSAMPLE);
    auto counts = count_labels(yb);
    EXPECT_EQ(counts[0.0f], counts[1.0f]);
    EXPECT_EQ(static_cast<size_t>(yb.size()), Xb.size());
}

TEST(test_balance, undersample_equalizes)
{
    auto X = make_dataset({10, 20, 30, 200}, 4);
    auto y = make_prob({0, 0, 0, 1});

    auto [Xb, yb] = ivc::student::balance(X, y, ivc::student::UNDERSAMPLE);
    auto counts = count_labels(yb);
    EXPECT_EQ(counts[0.0f], counts[1.0f]);
    EXPECT_EQ(static_cast<size_t>(yb.size()), Xb.size());
}

TEST(test_balance, smote_equalizes)
{
    auto X = make_dataset({10, 20, 30, 200}, 4);
    auto y = make_prob({0, 0, 0, 1});

    auto [Xb, yb] = ivc::student::balance(X, y, ivc::student::SMOTE);
    auto counts = count_labels(yb);
    EXPECT_EQ(counts[0.0f], counts[1.0f]);
    EXPECT_EQ(static_cast<size_t>(yb.size()), Xb.size());
}

TEST(test_balance, smote_labels_are_known_classes)
{
    auto X = make_dataset({10, 20, 30, 200}, 4);
    auto y = make_prob({0, 0, 0, 1});

    auto [Xb, yb] = ivc::student::balance(X, y, ivc::student::SMOTE);
    for (Eigen::Index i = 0; i < yb.size(); ++i)
        EXPECT_TRUE(yb(i) == 0.0f || yb(i) == 1.0f) << "Unknown label: " << yb(i);
}

TEST(test_balance, smote_pixels_in_range)
{
    auto X = make_dataset({10, 20, 30, 200}, 4);
    auto y = make_prob({0, 0, 0, 1});

    auto [Xb, yb] = ivc::student::balance(X, y, ivc::student::SMOTE);
    for (const auto& img : Xb)
        for (int r = 0; r < img.rows(); ++r)
            for (int c = 0; c < img.cols(); ++c)
                EXPECT_LE(static_cast<int>(img(r, c)), 255);
}

TEST(test_balance, dataset_size_matches_labels)
{
    auto X = make_dataset({0, 0, 0, 255}, 4);
    auto y = make_prob({0, 0, 0, 1});

    for (auto bt : {ivc::student::OVERSAMPLE,
                    ivc::student::UNDERSAMPLE,
                    ivc::student::SMOTE})
    {
        auto [Xb, yb] = ivc::student::balance(X, y, bt);
        EXPECT_EQ(Xb.size(), static_cast<size_t>(yb.size()));
    }
}
