#pragma once
#include <string>
#include <fstream>
#include <sadl/model.h>

inline sadl::layers::TensorInternalType::Type getModelType(const std::string &filename)
{
  const std::string MAGICNUMBER = "SADL0002";
  std::ifstream     file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "[ERROR] No file " << filename << std::endl;
    exit(-1);
  }
  char         magic[9];
  file.read(magic, 8);
  magic[8]       = '\0';
  std::string magic_s = magic;
  if (magic_s != MAGICNUMBER)
  {
    std::cerr << "[ERROR] Pb reading model: wrong magic " << magic_s << std::endl;
    exit(-1);
  }

  int8_t x = 0;
  file.read((char *) &x, sizeof(int8_t));
  return (sadl::layers::TensorInternalType::Type) x;
}

