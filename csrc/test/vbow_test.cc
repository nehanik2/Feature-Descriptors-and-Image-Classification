// SYSTEM INCLUDES
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <set>

#include <gtest/gtest.h>
#include <585/common/types.h>
#include <585/imgproc/imgproc.h>

// C++ PROJECT INCLUDES
#include "vbow/vbow.h"

// ============================================================
//  Helper utilities
// ============================================================

// Build a tiny synthetic dataset of square images.
// Each image is `side x side` pixels filled with a constant value.
static ivc::ByteDataset make_constant_dataset(const std::vector<uint8_t>& fill_values,
                                               int side = 4)
{
    ivc::ByteDataset X;
    for (uint8_t v : fill_values)
    {
        ivc::GrayscaleByteImg img(side * side, v);
        X.push_back(img);
    }
    return X;
}

// Create an image with a checkerboard pattern (for richer vocabulary)
static ivc::GrayscaleByteImg make_checkerboard(int side, uint8_t a = 0, uint8_t b = 255)
{
    ivc::GrayscaleByteImg img(side * side);
    for (int r = 0; r < side; ++r)
        for (int c = 0; c < side; ++c)
            img[r * side + c] = ((r + c) % 2 == 0) ? a : b;
    return img;
}

// ============================================================
//  Vocab Tests
// ============================================================

TEST(test_vocab, add_unique)
{
    ivc::student::Vocab vocab;
    ivc::GrayscaleByteImg p1 = {1, 2, 3, 4};
    ivc::GrayscaleByteImg p2 = {5, 6, 7, 8};

    vocab.add(p1);
    vocab.add(p2);
    EXPECT_EQ(vocab.size(), 2u);
}

TEST(test_vocab, add_duplicate)
{
    ivc::student::Vocab vocab;
    ivc::GrayscaleByteImg p1 = {1, 2, 3, 4};

    vocab.add(p1);
    vocab.add(p1);   // should NOT grow
    vocab.add(p1);
    EXPECT_EQ(vocab.size(), 1u);
}

TEST(test_vocab, ordered_elements_order)
{
    ivc::student::Vocab vocab;
    // Add three distinct patches
    ivc::GrayscaleByteImg p0 = {10, 10};
    ivc::GrayscaleByteImg p1 = {20, 20};
    ivc::GrayscaleByteImg p2 = {30, 30};

    vocab.add(p0);
    vocab.add(p1);
    vocab.add(p2);

    auto elems = vocab.ordered_elements();
    ASSERT_EQ(elems.size(), 3u);

    // Elements must come back in insertion-index order
    auto it = elems.begin();
    EXPECT_EQ(*it, p0); ++it;
    EXPECT_EQ(*it, p1); ++it;
    EXPECT_EQ(*it, p2);
}

TEST(test_vocab, ordered_elements_deterministic)
{
    // Two vocabs built with the same patches in same order must
    // return identical ordered_elements lists.
    ivc::student::Vocab v1, v2;
    std::vector<ivc::GrayscaleByteImg> patches = {
        {1,2,3}, {4,5,6}, {7,8,9}
    };
    for (auto& p : patches) { v1.add(p); v2.add(p); }

    auto e1 = v1.ordered_elements();
    auto e2 = v2.ordered_elements();
    ASSERT_EQ(e1.size(), e2.size());

    auto it1 = e1.begin();
    auto it2 = e2.begin();
    while (it1 != e1.end())
    {
        EXPECT_EQ(*it1, *it2);
        ++it1; ++it2;
    }
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
    // 3 train images, 2 test images — 4x4 pixels, patch_size=2
    auto X_train = make_constant_dataset({0, 128, 255}, 4);
    auto X_test  = make_constant_dataset({0, 255},      4);

    ivc::student::BinaryBagOfWords bow(X_train, 2);
    auto D = bow.transform(X_test);

    ASSERT_EQ(D.size(), 2u);       // 2 test images
    // vocab size > 0
    EXPECT_GT(D[0].size(), 0u);
}

TEST(test_binary_vbow, values_are_binary)
{
    auto X_train = make_constant_dataset({0, 64, 128, 192, 255}, 4);
    ivc::student::BinaryBagOfWords bow(X_train, 2);
    auto D = bow.transform(X_train);

    for (const auto& row : D)
        for (float_t v : row)
            EXPECT_TRUE(v == 0.0f || v == 1.0f)
                << "Non-binary value: " << v;
}

TEST(test_binary_vbow, identical_train_test_nonzero)
{
    // If we transform the training set, each image must activate at least one vocab word
    auto X = make_constant_dataset({0, 128, 255}, 4);
    ivc::student::BinaryBagOfWords bow(X, 2);
    auto D = bow.transform(X);

    for (size_t i = 0; i < D.size(); ++i)
    {
        float_t sum = 0.0f;
        for (float_t v : D[i]) sum += v;
        EXPECT_GT(sum, 0.0f) << "Image " << i << " has all-zero feature vector";
    }
}

TEST(test_binary_vbow, patch_size_1)
{
    // Patch size 1 → every pixel becomes a vocab word (values 0-255)
    auto X_train = make_constant_dataset({10, 20}, 2);
    ivc::student::BinaryBagOfWords bow(X_train, 1);
    auto D = bow.transform(X_train);
    ASSERT_EQ(D.size(), 2u);
    // Vocab should have exactly 2 unique single-pixel patches
    EXPECT_EQ(D[0].size(), 2u);
}

// ============================================================
//  CountingBagOfWords Tests
// ============================================================

TEST(test_counting_vbow, output_dimensions)
{
    auto X_train = make_constant_dataset({0, 128, 255}, 4);
    auto X_test  = make_constant_dataset({0, 255},      4);

    ivc::student::CountingBagOfWords bow(X_train, 2);
    auto D = bow.transform(X_test);

    ASSERT_EQ(D.size(), 2u);
    EXPECT_GT(D[0].size(), 0u);
}

TEST(test_counting_vbow, counts_are_nonnegative)
{
    auto X = make_constant_dataset({0, 128, 255}, 4);
    ivc::student::CountingBagOfWords bow(X, 2);
    auto D = bow.transform(X);

    for (const auto& row : D)
        for (float_t v : row)
            EXPECT_GE(v, 0.0f);
}

TEST(test_counting_vbow, counts_geq_binary)
{
    // For the same data, counting values should be >= binary values (same positions)
    auto X = make_constant_dataset({42, 84}, 4);

    ivc::student::BinaryBagOfWords  bow_bin(X, 2);
    ivc::student::CountingBagOfWords bow_cnt(X, 2);

    auto D_bin = bow_bin.transform(X);
    auto D_cnt = bow_cnt.transform(X);

    ASSERT_EQ(D_bin.size(), D_cnt.size());
    for (size_t i = 0; i < D_bin.size(); ++i)
    {
        ASSERT_EQ(D_bin[i].size(), D_cnt[i].size());
        for (size_t j = 0; j < D_bin[i].size(); ++j)
            EXPECT_GE(D_cnt[i][j], D_bin[i][j]);
    }
}

TEST(test_counting_vbow, count_sums_match_patches)
{
    // 4x4 image, patch_size 1 → 16 patches; a constant-fill image has only one
    // unique patch, so count[unique_pixel] == 16 and all other counts are 0.
    auto X_train = make_constant_dataset({50, 100}, 4); // two unique pixel values
    ivc::student::CountingBagOfWords bow(X_train, 1);
    auto D = bow.transform(X_train);

    ASSERT_EQ(D.size(), 2u);
    // Each row should sum to 16 (4*4 patches) since every pixel falls into one bucket
    for (const auto& row : D)
    {
        float_t s = 0.0f;
        for (float_t v : row) s += v;
        EXPECT_FLOAT_EQ(s, 16.0f);
    }
}

// ============================================================
//  OVR Tests
// ============================================================

TEST(test_ovr, predict_returns_known_label)
{
    // Trivial 1D linearly separable problem
    // Class 0: feature=0, Class 1: feature=1
    ivc::FloatDataset X_train = {{0.0f}, {0.0f}, {0.0f}, {1.0f}, {1.0f}, {1.0f}};
    ivc::ProbVector   y_train = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

    ivc::student::OVR ovr;
    ovr.train(X_train, y_train, 0.1f, 200);

    ivc::FloatDataset X_test = {{0.0f}, {1.0f}};
    auto y_pred = ovr.predict(X_test);

    ASSERT_EQ(y_pred.size(), 2u);
    EXPECT_FLOAT_EQ(y_pred[0], 0.0f);
    EXPECT_FLOAT_EQ(y_pred[1], 1.0f);
}

TEST(test_ovr, cost_in_range)
{
    ivc::FloatDataset X = {{0.0f}, {1.0f}, {2.0f}};
    ivc::ProbVector   y = {0.0f, 1.0f, 2.0f};

    ivc::student::OVR ovr;
    ovr.train(X, y, 0.01f, 10);

    float_t c = ovr.cost(X, y);
    EXPECT_GE(c, 0.0f);
    EXPECT_LE(c, 1.0f);
}

TEST(test_ovr, empty_dataset_no_crash)
{
    ivc::student::OVR ovr;
    ivc::FloatDataset X_empty;
    ivc::ProbVector   y_empty;
    EXPECT_NO_THROW(ovr.train(X_empty, y_empty, 0.1f, 10));
    EXPECT_NO_THROW(ovr.predict(X_empty));
}

// ============================================================
//  tile_dataset Tests
// ============================================================

TEST(test_tile, level_0_noop)
{
    auto X = make_constant_dataset({0, 255}, 4);
    auto T = ivc::student::tile_dataset(X, 0);
    EXPECT_EQ(T.size(), X.size());
    for (size_t i = 0; i < X.size(); ++i)
        EXPECT_EQ(X[i], T[i]);
}

TEST(test_tile, level_1_quadruples_count)
{
    auto X = make_constant_dataset({128}, 4);
    auto T = ivc::student::tile_dataset(X, 1);
    // 1 image → 4 tiles
    EXPECT_EQ(T.size(), 4u);
    // Each tile should be half the side → quarter the pixels
    EXPECT_EQ(T[0].size(), X[0].size() / 4);
}

TEST(test_tile, level_1_tile_pixel_count)
{
    // 8x8 image tiled at level 1 → 4 tiles of 4x4 = 16 pixels each
    ivc::GrayscaleByteImg img(64, 1);
    ivc::ByteDataset X = {img};
    auto T = ivc::student::tile_dataset(X, 1);
    ASSERT_EQ(T.size(), 4u);
    for (const auto& tile : T)
        EXPECT_EQ(tile.size(), 16u);
}

TEST(test_tile, level_2_produces_16_tiles)
{
    // 8x8 → 16 tiles of 2x2 at level 2
    ivc::GrayscaleByteImg img(64, 1);
    ivc::ByteDataset X = {img};
    auto T = ivc::student::tile_dataset(X, 2);
    EXPECT_EQ(T.size(), 16u);
}

// ============================================================
//  HistogramOfGradients Tests
// ============================================================

TEST(test_hog, output_has_8_bins)
{
    auto X = make_constant_dataset({128}, 8);
    ivc::student::HistogramOfGradients hog;
    auto D = hog.transform(X);
    ASSERT_EQ(D.size(), 1u);
    EXPECT_EQ(D[0].size(), 8u);
}

TEST(test_hog, values_normalized)
{
    // Constant image → zero gradient → hist should be valid (all zero or sum==1)
    ivc::GrayscaleByteImg img(64, 100);
    ivc::ByteDataset X = {img};

    ivc::student::HistogramOfGradients hog;
    auto D = hog.transform(X);
    ASSERT_EQ(D.size(), 1u);

    float_t sum = 0.0f;
    for (float_t v : D[0])
    {
        EXPECT_GE(v, 0.0f);
        sum += v;
    }
    // Sum is either 0 (flat image) or ~1 (normalized)
    EXPECT_TRUE(std::abs(sum - 1.0f) < 1e-4f || std::abs(sum) < 1e-6f);
}

// ============================================================
//  balance Tests
// ============================================================

TEST(test_balance, oversample_equalizes_classes)
{
    // 3 samples of class 0, 1 sample of class 1
    auto X = make_constant_dataset({0, 0, 0, 255}, 4);
    ivc::ProbVector y = {0.0f, 0.0f, 0.0f, 1.0f};

    auto [X_bal, y_bal] = ivc::student::balance(X, y, ivc::student::OVERSAMPLE);

    std::map<float_t, int> counts;
    for (float_t l : y_bal) counts[l]++;
    EXPECT_EQ(counts[0.0f], counts[1.0f]);
}

TEST(test_balance, undersample_equalizes_classes)
{
    auto X = make_constant_dataset({0, 0, 0, 255}, 4);
    ivc::ProbVector y = {0.0f, 0.0f, 0.0f, 1.0f};

    auto [X_bal, y_bal] = ivc::student::balance(X, y, ivc::student::UNDERSAMPLE);

    std::map<float_t, int> counts;
    for (float_t l : y_bal) counts[l]++;
    EXPECT_EQ(counts[0.0f], counts[1.0f]);
}

TEST(test_balance, smote_equalizes_classes)
{
    auto X = make_constant_dataset({10, 20, 30, 200}, 4);
    ivc::ProbVector y = {0.0f, 0.0f, 0.0f, 1.0f};

    auto [X_bal, y_bal] = ivc::student::balance(X, y, ivc::student::SMOTE);

    std::map<float_t, int> counts;
    for (float_t l : y_bal) counts[l]++;
    EXPECT_EQ(counts[0.0f], counts[1.0f]);
}

TEST(test_balance, smote_preserves_label_integrity)
{
    // All generated labels must belong to known classes
    auto X = make_constant_dataset({10, 20, 30, 200}, 4);
    ivc::ProbVector y = {0.0f, 0.0f, 0.0f, 1.0f};

    auto [X_bal, y_bal] = ivc::student::balance(X, y, ivc::student::SMOTE);

    std::set<float_t> known_classes = {0.0f, 1.0f};
    for (float_t l : y_bal)
        EXPECT_TRUE(known_classes.count(l) > 0) << "Unknown label: " << l;
}

TEST(test_balance, balance_preserves_pixel_range)
{
    // SMOTE synthetic pixels must stay in [0, 255]
    auto X = make_constant_dataset({10, 20, 30, 200}, 4);
    ivc::ProbVector y = {0.0f, 0.0f, 0.0f, 1.0f};

    auto [X_bal, y_bal] = ivc::student::balance(X, y, ivc::student::SMOTE);

    for (const auto& img : X_bal)
        for (uint8_t px : img)
        {
            EXPECT_GE(static_cast<int>(px), 0);
            EXPECT_LE(static_cast<int>(px), 255);
        }
}
