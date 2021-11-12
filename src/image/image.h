#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

// forward declarations of jpeglib struct
struct jpeg_error_mgr;

class Image {
public:
  struct Pixel {
    friend class Image;

  public:
    size_t GetPixelSize() const { return size; };
    uint8_t operator[](size_t n) const { return data[n]; };

  private:
    Pixel(const uint8_t *ptr, size_t psize) : data(ptr), size(psize){};
    const uint8_t *data;
    size_t size;
  };

public:
  using ImageError = std::string;

  static constexpr double kRLumWeight = 0.2126;
  static constexpr double kGLumWeight = 0.7152;
  static constexpr double kBLumWeight = 0.0722;

  static std::variant<Image, ImageError> OpenImage(const std::string &filename);

  // Currently can only construct with an existing file.
  // Will throw if file cannot be loaded, or is in the wrong format,
  // or some other error is encountered.
  // explicit Image(const std::string &fileName);
  explicit Image(size_t width, size_t height, size_t pixelSize, int colorSpace);

  // We can construct from an existing image object. This allows us
  // to work on a copy (e.g. shrink then save) without affecting the
  // original we have in memory.
  Image(const Image &rhs);
  Image(Image &&);

  // But assigment and move operations are currently disallowed
  Image &operator=(const Image &) = delete;
  Image &operator=(Image &&) = delete;

  ~Image();

  // Will throw if file cannot be saved. If no
  // filename is supplied, writes to fileName supplied in load()
  // (if that was called, otherwise throws)
  // Quality's usable values are 0-100
  std::variant<std::monostate, ImageError> Save(const std::string &fileName,
                                                int quality = 95) const;

  // Mainly for testing, writes an uncompressed PPM file
  std::variant<std::monostate, ImageError>
  SavePpm(const std::string &fileName) const;

  size_t GetHeight() const { return m_height; }
  size_t GetWidth() const { return m_width; }
  size_t GetPixelSize() const { return m_pixelSize; }

  // Will return a vector of pixel components. The vector's
  // size will be 1 for monochrome or 3 for RGB.
  // Elements for the latter will be in order R, G, B.
  Pixel GetPixel(size_t x, size_t y) const;

  uint8_t GetLuminance(size_t x, size_t y) const;
  void AddLuminance(size_t x, size_t y, int value);

  uint8_t *GetPixelData(size_t x, size_t y);
  const uint8_t *GetPixelData(size_t x, size_t y) const;

private:
  std::vector<uint8_t> m_bitmapData;
  size_t m_width;
  size_t m_height;
  size_t m_pixelSize;
  int m_colourSpace;
};
