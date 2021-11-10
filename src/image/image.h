#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// forward declarations of jpeglib struct
struct jpeg_error_mgr;

class Image {
public:
  // Currently can only construct with an existing file.
  // Will throw if file cannot be loaded, or is in the wrong format,
  // or some other error is encountered.
  explicit Image(const std::string &fileName);

  // We can construct from an existing image object. This allows us
  // to work on a copy (e.g. shrink then save) without affecting the
  // original we have in memory.
  Image(const Image &rhs);

  // But assigment and move operations are currently disallowed
  Image &operator=(const Image &) = delete;
  Image(Image &&) = delete;
  Image &operator=(Image &&) = delete;

  ~Image();

  // Will throw if file cannot be saved. If no
  // filename is supplied, writes to fileName supplied in load()
  // (if that was called, otherwise throws)
  // Quality's usable values are 0-100
  void Save(const std::string &fileName, int quality = 95) const;

  // Mainly for testing, writes an uncompressed PPM file
  void SavePpm(const std::string &fileName) const;

  size_t GetHeight() const { return m_height; }
  size_t GetWidth() const { return m_width; }
  size_t GetPixelSize() const { return m_pixelSize; }

  // Will return a vector of pixel components. The vector's
  // size will be 1 for monochrome or 3 for RGB.
  // Elements for the latter will be in order R, G, B.
  std::vector<uint8_t> GetPixel(size_t x, size_t y) const;

  uint8_t *GetPixelData(size_t x, size_t y);

private:
  // Note that m_errorMgr is a shared ptr and will be shared
  // between objects if one copy constructs from another
  std::shared_ptr<::jpeg_error_mgr> m_errorMgr;
  std::vector<uint8_t> m_bitmapData;
  size_t m_width;
  size_t m_height;
  size_t m_pixelSize;
  int m_colourSpace;
};
