#include "image.h"

#include <jpeglib.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

Image::Image(const std::string &fileName) {
  // Creating a custom deleter for the decompressInfo pointer
  // to ensure ::jpeg_destroy_compress() gets called even if
  // we throw out of this function.
  auto dt = [](::jpeg_decompress_struct *ds) { ::jpeg_destroy_decompress(ds); };
  std::unique_ptr<::jpeg_decompress_struct, decltype(dt)> decompressInfo(
      new ::jpeg_decompress_struct, dt);

  // Note this is a shared pointer as we can share this
  // between objects which have copy constructed from each other
  m_errorMgr = std::make_shared<::jpeg_error_mgr>();

  // Using fopen here ( and in save() ) because libjpeg expects
  // a FILE pointer.
  // We store the FILE* in a unique_ptr so we can also use the custom
  // deleter here to ensure fclose() gets called even if we throw.
  auto fdt = [](FILE *fp) { fclose(fp); };
  std::unique_ptr<FILE, decltype(fdt)> infile(fopen(fileName.c_str(), "rb"),
                                              fdt);
  if (infile.get() == NULL) {
    throw std::runtime_error("Could not open " + fileName);
  }

  decompressInfo->err = ::jpeg_std_error(m_errorMgr.get());
  // Note this usage of a lambda to provide our own error handler
  // to libjpeg. If we do not supply a handler, and libjpeg hits
  // a problem, it just prints the error message and calls exit().
  m_errorMgr->error_exit = [](::j_common_ptr cinfo) {
    char jpegLastErrorMsg[JMSG_LENGTH_MAX];
    // Call the function pointer to get the error message
    (*(cinfo->err->format_message))(cinfo, jpegLastErrorMsg);
    throw std::runtime_error(jpegLastErrorMsg);
  };
  ::jpeg_create_decompress(decompressInfo.get());

  // Read the file:
  ::jpeg_stdio_src(decompressInfo.get(), infile.get());

  int rc = ::jpeg_read_header(decompressInfo.get(), TRUE);
  if (rc != 1) {
    throw std::runtime_error("File does not seem to be a normal JPEG");
  }
  ::jpeg_start_decompress(decompressInfo.get());

  m_width = decompressInfo->output_width;
  m_height = decompressInfo->output_height;
  m_pixelSize = decompressInfo->output_components;
  m_colourSpace = decompressInfo->out_color_space;

  size_t row_stride = m_width * m_pixelSize;

  m_bitmapData.clear();
  m_bitmapData.resize(row_stride * m_height);

  size_t line_start = 0;
  while (decompressInfo->output_scanline < m_height) {
    uint8_t *p = m_bitmapData.data() + line_start;
    ::jpeg_read_scanlines(decompressInfo.get(), &p, 1);
    line_start += row_stride;
  }
  ::jpeg_finish_decompress(decompressInfo.get());
}

// Copy constructor
Image::Image(const Image &rhs) {
  m_errorMgr = rhs.m_errorMgr;
  m_bitmapData = rhs.m_bitmapData;
  m_width = rhs.m_width;
  m_height = rhs.m_height;
  m_pixelSize = rhs.m_pixelSize;
  m_colourSpace = rhs.m_colourSpace;
}

Image::~Image() {}

void Image::Save(const std::string &fileName, int quality) const {
  if (quality < 0) {
    quality = 0;
  }
  if (quality > 100) {
    quality = 100;
  }
  FILE *outfile = fopen(fileName.c_str(), "wb");
  if (outfile == NULL) {
    throw std::runtime_error("Could not open " + fileName + " for writing");
  }
  // Creating a custom deleter for the compressInfo pointer
  // to ensure ::jpeg_destroy_compress() gets called even if
  // we throw out of this function.
  auto dt = [](::jpeg_compress_struct *cs) { ::jpeg_destroy_compress(cs); };
  std::unique_ptr<::jpeg_compress_struct, decltype(dt)> compressInfo(
      new ::jpeg_compress_struct, dt);
  ::jpeg_create_compress(compressInfo.get());
  ::jpeg_stdio_dest(compressInfo.get(), outfile);
  compressInfo->image_width = m_width;
  compressInfo->image_height = m_height;
  compressInfo->input_components = m_pixelSize;
  compressInfo->in_color_space = static_cast<::J_COLOR_SPACE>(m_colourSpace);
  compressInfo->err = ::jpeg_std_error(m_errorMgr.get());
  ::jpeg_set_defaults(compressInfo.get());
  ::jpeg_set_quality(compressInfo.get(), quality, TRUE);
  ::jpeg_start_compress(compressInfo.get(), TRUE);

  size_t line_start = 0;
  size_t row_stride = m_width * m_pixelSize;
  for (; line_start < m_bitmapData.size(); line_start += row_stride) {
    ::JSAMPROW rowPtr[1];
    // Casting const-ness away here because the jpeglib
    // call expects a non-const pointer. It presumably
    // doesn't modify our data.
    rowPtr[0] = const_cast<::JSAMPROW>(m_bitmapData.data() + line_start);
    ::jpeg_write_scanlines(compressInfo.get(), rowPtr, 1);
  }
  ::jpeg_finish_compress(compressInfo.get());
  fclose(outfile);
}

void Image::SavePpm(const std::string &fileName) const {
  std::ofstream ofs(fileName, std::ios::out | std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("Could not open " + fileName + " for saving");
  }
  // Write the header
  ofs << "P6 " << m_width << " " << m_height << " 255\n";
  ofs.write(reinterpret_cast<const char *>(m_bitmapData.data()),
            m_bitmapData.size());
  ofs.close();
}

std::vector<uint8_t> Image::GetPixel(size_t x, size_t y) const {
  // if (y >= m_bitmapData.size()) {
  //   throw std::out_of_range("Y value too large");
  // }
  // if (x >= m_bitmapData[0].size() / m_pixelSize) {
  //   throw std::out_of_range("X value too large");
  // }
  std::vector<uint8_t> vec;
  for (size_t n = 0; n < m_pixelSize; ++n) {
    vec.push_back(
        m_bitmapData[y * m_width * m_pixelSize + x * m_pixelSize + n]);
  }
  return vec;
}

uint8_t *Image::GetPixelData(size_t x, size_t y) {
  return &m_bitmapData[y * m_width * m_pixelSize + x * m_pixelSize];
}
