import sharp from "sharp";
const [,, src, dst, left, top, width, height, scale] = process.argv;
await sharp(src)
  .extract({ left: +left, top: +top, width: +width, height: +height })
  .resize({ width: Math.round(+width * +scale), kernel: "nearest" })
  .png()
  .toFile(dst);
console.log("wrote", dst);
