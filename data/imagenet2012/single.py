import tarfile
import os
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import torch
from torchvision import transforms
import io

# ---------- 配置 ----------
outer_tar = '/home/wangjzh/adam_optimizer/data/imagenet2012/train.tar'          # 或 'data/val.tar'
save_root = 'home/wangjzh/adam_optimizer/data/imagenet2012/train_preprocessed'  # 或 'data/val_preprocessed'
image_size = 224

# ---------- Transform ----------
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    normalize,
])

# ---------- 处理每个子tar ----------
def process_inner_tar(inner_bytes, class_name):
    try:
        with tarfile.open(fileobj=inner_bytes, mode='r') as inner_tar:
            members = inner_tar.getmembers()
            print(f"  ↪️  [{class_name}] contains {len(members)} files")
            for member in members:
                if not member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                f = inner_tar.extractfile(member)
                if f is None:
                    print(f"    ⚠️  Cannot extract file: {member.name}")
                    continue
                try:
                    img = Image.open(f).convert('RGB')
                    tensor = transform(img)
                except UnidentifiedImageError:
                    print(f"    ❌ Skipped bad image: {member.name}")
                    continue
                save_path = os.path.join(save_root, class_name,
                                         os.path.splitext(os.path.basename(member.name))[0] + '.pt')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(tensor, save_path)
    except Exception as e:
        print(f"  ❗ Error in processing inner tar for class [{class_name}]: {e}")

# ---------- 主程序 ----------
def main():
    print(f"🚀 Start processing: {outer_tar}")
    try:
        with tarfile.open(outer_tar, 'r') as outer:
            members = [m for m in outer.getmembers() if m.name.endswith('.tar')]
        print(f"📦 Found {len(members)} class tar files")
    except Exception as e:
        print(f"❗ Failed to open outer tar: {e}")
        return

    for member in tqdm(members):
        class_name = os.path.splitext(os.path.basename(member.name))[0]
        print(f"🔧 Processing: {class_name}")
        try:
            with tarfile.open(outer_tar, 'r') as outer:
                inner_f = outer.extractfile(member)
                if inner_f is None:
                    print(f"❌ Failed to extract: {member.name}")
                    continue
                inner_bytes = io.BytesIO(inner_f.read())
                process_inner_tar(inner_bytes, class_name)
            print(f"✅ Done: {class_name}")
        except Exception as e:
            print(f"❗ Error processing {member.name}: {e}")

    print("🏁 All done.")

if __name__ == '__main__':
    main()
