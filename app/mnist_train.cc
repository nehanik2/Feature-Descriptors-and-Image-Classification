// mnist_train.cc
// ============================================================
//  MNIST Training Script for Visual Bag of Words + OVR
//  CS585 – Image and Video Computing
//
//  Usage:
//    ./mnist_train <mnist_data_dir> [patch_size] [lr] [max_epochs] [bow_type]
//
//  bow_type: 0=binary (default), 1=counting, 2=hog
//
//  Example:
//    ./mnist_train /data/mnist 3 0.01 50 0
// ============================================================

// SYSTEM INCLUDES
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>

// FRAMEWORK INCLUDES
#include <585/common/types.h>
#include <585/vbow/vbow.h>      // LogReg, dataset types

// PROJECT INCLUDES
#include "vbow/vbow.h"


// ============================================================
//  MNIST Binary File Parsing
//  Ref: http://yann.lecun.com/exdb/mnist/
// ============================================================

static uint32_t read_be_uint32(std::ifstream& f)
{
    uint8_t buf[4];
    f.read(reinterpret_cast<char*>(buf), 4);
    return (static_cast<uint32_t>(buf[0]) << 24)
         | (static_cast<uint32_t>(buf[1]) << 16)
         | (static_cast<uint32_t>(buf[2]) <<  8)
         |  static_cast<uint32_t>(buf[3]);
}

static ivc::ByteDataset load_mnist_images(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t magic  = read_be_uint32(f);
    uint32_t num    = read_be_uint32(f);
    uint32_t rows   = read_be_uint32(f);
    uint32_t cols   = read_be_uint32(f);

    if (magic != 0x00000803)
        throw std::runtime_error("Bad MNIST image magic in " + path);

    ivc::ByteDataset X;
    X.reserve(num);
    const size_t px = rows * cols;

    for (uint32_t i = 0; i < num; ++i)
    {
        ivc::GrayscaleByteImg img(px);
        f.read(reinterpret_cast<char*>(img.data()), px);
        X.push_back(std::move(img));
    }
    return X;
}

static ivc::ProbVector load_mnist_labels(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t magic = read_be_uint32(f);
    uint32_t num   = read_be_uint32(f);

    if (magic != 0x00000801)
        throw std::runtime_error("Bad MNIST label magic in " + path);

    ivc::ProbVector y(num);
    for (uint32_t i = 0; i < num; ++i)
    {
        uint8_t lbl;
        f.read(reinterpret_cast<char*>(&lbl), 1);
        y[i] = static_cast<float_t>(lbl);
    }
    return y;
}

// ============================================================
//  Evaluation
// ============================================================

static float_t accuracy(const ivc::ProbVector& y_pred,
                         const ivc::ProbVector& y_gt)
{
    size_t correct = 0;
    for (size_t i = 0; i < y_gt.size(); ++i)
        if (y_pred[i] == y_gt[i]) ++correct;
    return static_cast<float_t>(correct) / static_cast<float_t>(y_gt.size());
}

// ============================================================
//  Print helpers
// ============================================================

static void print_separator()
{
    std::cout << std::string(60, '=') << "\n";
}

static void print_config(int patch_size, float_t lr, int epochs, int bow_type)
{
    print_separator();
    std::cout << "  CS585 – MNIST Visual Bag of Words Training\n";
    print_separator();
    std::cout << "  patch_size : " << patch_size << "\n"
              << "  lr         : " << lr         << "\n"
              << "  max_epochs : " << epochs     << "\n"
              << "  bow_type   : " << (bow_type == 0 ? "BinaryBoW" :
                                       bow_type == 1 ? "CountingBoW" : "HOG") << "\n";
    print_separator();
}

// ============================================================
//  main
// ============================================================

int main(int argc, char** argv)
{
    // ---- Parse arguments ----
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <mnist_data_dir> [patch_size=3] [lr=0.01] [max_epochs=50] [bow_type=0]\n";
        return 1;
    }

    std::string data_dir  = argv[1];
    int    patch_size     = (argc > 2) ? std::stoi(argv[2]) : 3;
    float_t lr            = (argc > 3) ? std::stof(argv[3]) : 0.01f;
    int    max_epochs     = (argc > 4) ? std::stoi(argv[4]) : 50;
    int    bow_type       = (argc > 5) ? std::stoi(argv[5]) : 0;  // 0=binary, 1=counting, 2=hog

    print_config(patch_size, lr, max_epochs, bow_type);

    // ---- Load MNIST ----
    auto t0 = std::chrono::steady_clock::now();

    std::cout << "Loading MNIST from: " << data_dir << " ...\n";

    ivc::ByteDataset X_train = load_mnist_images(data_dir + "/train-images-idx3-ubyte");
    ivc::ProbVector  y_train = load_mnist_labels(data_dir + "/train-labels-idx1-ubyte");
    ivc::ByteDataset X_test  = load_mnist_images(data_dir + "/t10k-images-idx3-ubyte");
    ivc::ProbVector  y_test  = load_mnist_labels(data_dir + "/t10k-labels-idx1-ubyte");

    std::cout << "  Train: " << X_train.size() << " images\n"
              << "  Test : " << X_test.size()  << " images\n";

    auto t1 = std::chrono::steady_clock::now();
    std::cout << "  Loaded in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()
              << " ms\n";

    // ---- Feature Extraction ----
    ivc::FloatDataset F_train, F_test;

    if (bow_type == 2)
    {
        // HOG with level-1 tiling
        std::cout << "Extracting HOG features (tiling level=1) ...\n";
        auto tiled_train = ivc::student::tile_dataset(X_train, 1);
        auto tiled_test  = ivc::student::tile_dataset(X_test,  1);

        ivc::student::HistogramOfGradients hog;
        auto F_tiled_train = hog.transform(tiled_train);
        auto F_tiled_test  = hog.transform(tiled_test);

        // Reassemble: each original image now has 4 tiles of 8 bins → concat into 32-dim
        const size_t tiles_per_img = 4;
        const size_t hog_dim = 8;
        F_train.resize(X_train.size(), ivc::FloatRow(tiles_per_img * hog_dim, 0.0f));
        F_test .resize(X_test .size(), ivc::FloatRow(tiles_per_img * hog_dim, 0.0f));

        for (size_t i = 0; i < X_train.size(); ++i)
            for (size_t t = 0; t < tiles_per_img; ++t)
                for (size_t b = 0; b < hog_dim; ++b)
                    F_train[i][t * hog_dim + b] = F_tiled_train[i * tiles_per_img + t][b];

        for (size_t i = 0; i < X_test.size(); ++i)
            for (size_t t = 0; t < tiles_per_img; ++t)
                for (size_t b = 0; b < hog_dim; ++b)
                    F_test[i][t * hog_dim + b] = F_tiled_test[i * tiles_per_img + t][b];
    }
    else
    {
        auto t_feat0 = std::chrono::steady_clock::now();
        if (bow_type == 0)
        {
            std::cout << "Building BinaryBagOfWords (patch_size=" << patch_size << ") ...\n";
            ivc::student::BinaryBagOfWords bow(X_train, patch_size);
            std::cout << "  Vocabulary size: " << bow.vocab_size() << "\n";
            std::cout << "Transforming train set ...\n";
            F_train = bow.transform(X_train);
            std::cout << "Transforming test set  ...\n";
            F_test  = bow.transform(X_test);
        }
        else
        {
            std::cout << "Building CountingBagOfWords (patch_size=" << patch_size << ") ...\n";
            ivc::student::CountingBagOfWords bow(X_train, patch_size);
            std::cout << "  Vocabulary size: " << bow.vocab_size() << "\n";
            std::cout << "Transforming train set ...\n";
            F_train = bow.transform(X_train);
            std::cout << "Transforming test set  ...\n";
            F_test  = bow.transform(X_test);
        }
        auto t_feat1 = std::chrono::steady_clock::now();
        std::cout << "  Feature dim  : " << (F_train.empty() ? 0 : F_train[0].size()) << "\n"
                  << "  Feature time : "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_feat1-t_feat0).count()
                  << " ms\n";
    }

    // ---- Train OVR ----
    std::cout << "Training OVR classifier ...\n";
    auto t_train0 = std::chrono::steady_clock::now();

    ivc::student::OVR ovr;
    ovr.train(F_train, y_train, lr, static_cast<size_t>(max_epochs));

    auto t_train1 = std::chrono::steady_clock::now();
    std::cout << "  Train time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_train1-t_train0).count()
              << " ms\n";

    // ---- Evaluate ----
    std::cout << "Evaluating ...\n";
    auto y_pred_train = ovr.predict(F_train);
    auto y_pred_test  = ovr.predict(F_test);

    float_t train_acc = accuracy(y_pred_train, y_train);
    float_t test_acc  = accuracy(y_pred_test,  y_test);

    print_separator();
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Train Accuracy : " << train_acc * 100.0f << " %\n";
    std::cout << "  Test  Accuracy : " << test_acc  * 100.0f << " %\n";
    print_separator();

    // ---- Per-class breakdown ----
    std::cout << "\nPer-class accuracy on test set:\n";
    std::map<int, int> class_correct, class_total;
    for (size_t i = 0; i < y_test.size(); ++i)
    {
        int gt = static_cast<int>(y_test[i]);
        class_total[gt]++;
        if (y_pred_test[i] == y_test[i]) class_correct[gt]++;
    }
    for (auto& kv : class_total)
    {
        int cls = kv.first;
        float_t acc = static_cast<float_t>(class_correct[cls]) / kv.second;
        std::cout << "  Digit " << cls << " : "
                  << std::setw(6) << acc * 100.0f << " %"
                  << "  (" << class_correct[cls] << "/" << kv.second << ")\n";
    }
    print_separator();

    return 0;
}
