#include "image.h"

#include <jpeglib.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <stdexcept>
#include <string>
#include <vector>

std::variant<Image, Image::ImageError>
Image::OpenImage(const std::string &fileName) {
  // Creating a custom deleter for the decompressInfo pointer
  // to ensure ::jpeg_destroy_compress() gets called even if
  // we throw out of this function.
  auto dt = [](::jpeg_decompress_struct *ds) { ::jpeg_destroy_decompress(ds); };
  std::unique_ptr<::jpeg_decompress_struct, decltype(dt)> decompressInfo(
      new ::jpeg_decompress_struct, dt);

  // Note this is a shared pointer as we can share this
  // between objects which have copy constructed from each other
  std::unique_ptr<::jpeg_error_mgr> m_errorMgr =
      std::make_unique<::jpeg_error_mgr>();

  // Using fopen here ( and in save() ) because libjpeg expects
  // a FILE pointer.
  // We store the FILE* in a unique_ptr so we can also use the custom
  // deleter here to ensure fclose() gets called even if we throw.
  auto fdt = [](FILE *fp) { fclose(fp); };
  std::unique_ptr<FILE, decltype(fdt)> infile(fopen(fileName.c_str(), "rb"),
                                              fdt);
  if (infile.get() == nullptr) {
    return ImageError("Could not open " + fileName);
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

  try {
    ::jpeg_create_decompress(decompressInfo.get());

    // Read the file:
    ::jpeg_stdio_src(decompressInfo.get(), infile.get());

    int rc = ::jpeg_read_header(decompressInfo.get(), TRUE);
    if (rc != 1) {
      return ImageError("File does not seem to be a normal JPEG");
    }
    ::jpeg_start_decompress(decompressInfo.get());

    size_t width = decompressInfo->output_width;
    size_t height = decompressInfo->output_height;
    size_t pixelSize = decompressInfo->output_components;
    size_t colourSpace = decompressInfo->out_color_space;

    size_t row_stride = width * pixelSize;

    Image image(width, height, pixelSize, colourSpace);

    size_t line_start = 0;
    while (decompressInfo->output_scanline < height) {
      uint8_t *p = image.m_bitmapData.data() + line_start;
      ::jpeg_read_scanlines(decompressInfo.get(), &p, 1);
      line_start += row_stride;
    }
    ::jpeg_finish_decompress(decompressInfo.get());
    return image;
  } catch (std::runtime_error err) {
    return ImageError(err.what());
  }
}

Image::Image(size_t width, size_t height, size_t pixelSize, int colorSpace)
    : m_width(width), m_height(height), m_pixelSize(pixelSize),
      m_colourSpace(colorSpace), m_bitmapData(width * height * pixelSize, 0),
      m_hsvData(width * height * pixelSize, 0) {}

// Image::Image(const std::string &fileName) {
//   // Creating a custom deleter for the decompressInfo pointer
//   // to ensure ::jpeg_destroy_compress() gets called even if
//   // we throw out of this function.
//   auto dt = [](::jpeg_decompress_struct *ds) { ::jpeg_destroy_decompress(ds);
//   }; std::unique_ptr<::jpeg_decompress_struct, decltype(dt)> decompressInfo(
//       new ::jpeg_decompress_struct, dt);

//   // Note this is a shared pointer as we can share this
//   // between objects which have copy constructed from each other
//   m_errorMgr = std::make_shared<::jpeg_error_mgr>();

//   // Using fopen here ( and in save() ) because libjpeg expects
//   // a FILE pointer.
//   // We store the FILE* in a unique_ptr so we can also use the custom
//   // deleter here to ensure fclose() gets called even if we throw.
//   auto fdt = [](FILE *fp) { fclose(fp); };
//   std::unique_ptr<FILE, decltype(fdt)> infile(fopen(fileName.c_str(), "rb"),
//                                               fdt);
//   if (infile.get() == NULL) {
//     throw std::runtime_error("Could not open " + fileName);
//   }

//   decompressInfo->err = ::jpeg_std_error(m_errorMgr.get());
//   // Note this usage of a lambda to provide our own error handler
//   // to libjpeg. If we do not supply a handler, and libjpeg hits
//   // a problem, it just prints the error message and calls exit().
//   m_errorMgr->error_exit = [](::j_common_ptr cinfo) {
//     char jpegLastErrorMsg[JMSG_LENGTH_MAX];
//     // Call the function pointer to get the error message
//     (*(cinfo->err->format_message))(cinfo, jpegLastErrorMsg);
//     throw std::runtime_error(jpegLastErrorMsg);
//   };
//   ::jpeg_create_decompress(decompressInfo.get());

//   // Read the file:
//   ::jpeg_stdio_src(decompressInfo.get(), infile.get());

//   int rc = ::jpeg_read_header(decompressInfo.get(), TRUE);
//   if (rc != 1) {
//     throw std::runtime_error("File does not seem to be a normal JPEG");
//   }
//   ::jpeg_start_decompress(decompressInfo.get());

//   m_width = decompressInfo->output_width;
//   m_height = decompressInfo->output_height;
//   m_pixelSize = decompressInfo->output_components;
//   m_colourSpace = decompressInfo->out_color_space;

//   size_t row_stride = m_width * m_pixelSize;

//   m_bitmapData.clear();
//   m_bitmapData.resize(row_stride * m_height);

//   size_t line_start = 0;
//   while (decompressInfo->output_scanline < m_height) {
//     uint8_t *p = m_bitmapData.data() + line_start;
//     ::jpeg_read_scanlines(decompressInfo.get(), &p, 1);
//     line_start += row_stride;
//   }
//   ::jpeg_finish_decompress(decompressInfo.get());
// }

// Copy constructor
Image::Image(const Image &rhs) {
  m_bitmapData = rhs.m_bitmapData;
  m_hsvData = rhs.m_hsvData;
  m_width = rhs.m_width;
  m_height = rhs.m_height;
  m_pixelSize = rhs.m_pixelSize;
  m_colourSpace = rhs.m_colourSpace;
}

Image::Image(Image &&rhs) {
  m_bitmapData = std::move(rhs.m_bitmapData);
  m_hsvData = std::move(rhs.m_hsvData);
  m_width = rhs.m_width;
  m_height = rhs.m_height;
  m_pixelSize = rhs.m_pixelSize;
  m_colourSpace = rhs.m_colourSpace;

  rhs.m_width = 0;
  rhs.m_height = 0;
  rhs.m_pixelSize = 0;
  rhs.m_colourSpace = 0;
}

Image::~Image() {}

std::variant<std::monostate, Image::ImageError>
Image::Save(const std::string &fileName, int quality) const {
  if (quality < 0) {
    quality = 0;
  }
  if (quality > 100) {
    quality = 100;
  }
  FILE *outfile = fopen(fileName.c_str(), "wb");
  if (outfile == nullptr) {
    return ImageError("Could not open " + fileName + " for writing");
  }

  std::unique_ptr<::jpeg_error_mgr> m_errorMgr =
      std::make_unique<::jpeg_error_mgr>();
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
  return {};
}

std::variant<std::monostate, Image::ImageError>
Image::SavePpm(const std::string &fileName) const {
  std::ofstream ofs(fileName, std::ios::out | std::ios::binary);
  if (!ofs) {
    return ImageError("Could not open " + fileName + " for writing");
  }
  // Write the header
  ofs << "P6 " << m_width << " " << m_height << " 255\n";
  ofs.write(reinterpret_cast<const char *>(m_bitmapData.data()),
            m_bitmapData.size());
  ofs.close();
  return {};
}

Image::Pixel Image::GetPixel(size_t x, size_t y) const {
  return Pixel(GetPixelData(x, y), m_pixelSize);
}

uint8_t *Image::GetPixelData(size_t x, size_t y) {
  return &m_bitmapData[y * m_width * m_pixelSize + x * m_pixelSize];
}

const uint8_t *Image::GetPixelData(size_t x, size_t y) const {
  return m_bitmapData.data() + (y * m_width * m_pixelSize + x * m_pixelSize);
}

uint8_t Image::GetLuminance(size_t x, size_t y) const {
  const uint8_t *p = GetPixelData(x, y);
  double r = static_cast<double>(p[0]);
  double g = static_cast<double>(p[1]);
  double b = static_cast<double>(p[2]);

  return static_cast<uint8_t>(r * kRLumWeight + g * kGLumWeight +
                              b * kBLumWeight);
}

double *Image::GetHSVData(size_t x, size_t y) {
  return &m_hsvData[y * m_width * m_pixelSize + x * m_pixelSize];
}

const double *Image::GetHSV(size_t x, size_t y) const {
  return m_hsvData.data() + (y * m_width * m_pixelSize + x * m_pixelSize);
}

double *Image::GetValueData(size_t x, size_t y) {
  double *hsv = GetHSVData(x, y);
  return &hsv[2];
}

const double Image::GetValue(size_t x, size_t y) const {
  const double *hsv = GetHSV(x, y);
  return hsv[2];
}

void Image::RGB2HSV(size_t x, size_t y) {
  const uint8_t *p = GetPixelData(x, y);
  double r = static_cast<double>(p[0]) / 255.0;
  double g = static_cast<double>(p[1]) / 255.0;
  double b = static_cast<double>(p[2]) / 255.0;

  double cmax = std::max(r, std::max(g, b));
  double cmin = std::min(r, std::min(g, b));
  double diff = cmax - cmin;
  double h = -1.0, s = -1.0;

  if (cmax == cmin) {
    h = 0.0;
  } else if (cmax == r) {
    if (g >= b) {
      h = 60.0 * ((g - b) / diff) + 0.0;
    } else {
      h = 60.0 * ((g - b) / diff) + 360.0;
    }
  } else if (cmax == g) {
    h = 60.0 * ((b - r) / diff) + 120.0;
  } else if (cmax == b) {
    h = 60.0 * ((r - g) / diff) + 240.0;
  }

  if (cmax == 0) {
    s = 0.0;
  } else {
    s = diff / cmax;
  }

  double v = cmax;
  double *hsvx = GetHSVData(x, y);
  hsvx[0] = h;
  hsvx[1] = s;
  hsvx[2] = v;
}

void Image::HSV2RGB(size_t x, size_t y) {
  const double *hsv = GetHSV(x, y);
  double h = hsv[0];
  double s = hsv[1];
  double v = hsv[2];

  int i = (int)floor(h / 60.0) % 6;
  double f = h / 60.0 - (double)i;
  double p = v * (1.0 - s);
  double q = v * (1.0 - f * s);
  double t = v * (1.0 - (1.0 - f) * s);
  double r, g, b;

  switch (i) {
  case 0:
    r = v, g = t, b = p;
    break;
  case 1:
    r = q, g = v, b = p;
    break;
  case 2:
    r = p, g = v, b = t;
    break;
  case 3:
    r = p, g = q, b = v;
    break;
  case 4:
    r = t, g = p, b = v;
    break;
  case 5:
    r = v, g = p, b = q;
    break;
  default:
    break;
  }

  uint8_t *px = GetPixelData(x, y);
  px[0] = static_cast<double>(r * 255);
  px[1] = static_cast<double>(g * 255);
  px[2] = static_cast<double>(b * 255);
}

void Image::AddLuminance(size_t x, size_t y, int value) {
  uint8_t *p = GetPixelData(x, y);
  double dr = static_cast<double>(value) * kRLumWeight;
  double dg = static_cast<double>(value) * kGLumWeight;
  double db = static_cast<double>(value) * kBLumWeight;
}
