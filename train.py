import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep

import models
import datasets

from medicalnet_model import generate_model, generate_model_swin

def get_3d_sincos_pos_embed(D, H, W, dim, device):
    def sincos_embedding(pos, dim_half):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_half, 2).float() / dim_half)).to(device)
        sinusoid_inp = torch.einsum("i,j->ij", pos.float(), inv_freq)
        emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return emb

    d_pos = torch.arange(D, device=device)
    h_pos = torch.arange(H, device=device)
    w_pos = torch.arange(W, device=device)

    dim_each = dim // 3
    d_emb = sincos_embedding(d_pos, dim_each)
    h_emb = sincos_embedding(h_pos, dim_each)
    w_emb = sincos_embedding(w_pos, dim_each)

    pos_embed = torch.zeros((D, H, W, dim), device=device)
    for d in range(D):
        for h in range(H):
            for w in range(W):
                pos_embed[d, h, w] = torch.cat([d_emb[d], h_emb[h], w_emb[w]], dim=-1)

    return pos_embed.view(-1, dim)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.backbone == "resnet":
        classifier, parameters = generate_model(model_name=args.model_name,num_seg_classes=3, phase='train', pretrain_path=args.checkpoint_pretrain)
        if args.checkpoint_pretrain:
            print("Use pretrained model")
            params = [{ 'params': parameters['base_parameters'], 'lr': args.lr }, { 'params': parameters['new_parameters'], 'lr': args.lr*100 }]
            optimizer = torch.optim.Adam(params, weight_decay=1e-3)   
        else:
            print("Train from scratch")
            optimizer = torch.optim.Adam(parameters, weight_decay=1e-3, lr=args.lr)     
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif args.backbone == "swin":
        classifier, parameters = generate_model_swin(pretrain_path=args.checkpoint_pretrain)
        if args.checkpoint_pretrain:
            print("Use pretrained model")
            # params = [{ 'params': parameters['base_parameters'], 'lr': args.lr }, { 'params': parameters['new_parameters'], 'lr': args.lr*100 }]
            optimizer = torch.optim.Adam(parameters, weight_decay=1e-3, lr=args.lr) 
        else:
            optimizer = torch.optim.Adam(parameters, weight_decay=1e-3, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.CrossEntropyLoss()

    train_dataset = datasets.MyDataSet.MyDataset(json_path=args.train_json, image_dir=args.train_image_dir)
    valid_dataset = datasets.MyDataSet.MyDataset(json_path=args.valid_json, image_dir=args.valid_image_dir)

    print(f"Length of training dataset: {len(train_dataset)}")
    print(f"Length of validation dataset: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        for i, group in enumerate(optimizer.param_groups):
            print(f"Epoch {epoch + 1} - Learning Rate Group {i}: {group['lr']:.6f}")
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits = classifier(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        scheduler.step()
        # Validation
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                logits = classifier(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_avg_loss = val_loss / val_total
        val_acc = val_correct / val_total
        if (epoch + 1) % 10 == 0:
            if args.checkpoint_pretrain:
                torch.save(classifier.state_dict(), f"/scratch/user/baileyyeah/Alzheimer_my/results/{args.backbone}/model_{args.model_name}_bs{args.batch_size}_epoch_{epoch+1}.pth")
            else:
                torch.save(classifier.state_dict(), f"/scratch/user/baileyyeah/Alzheimer_my/results/{args.backbone}/scratch/model_{args.model_name}_bs{args.batch_size}_epoch_{epoch+1}.pth")
            print(f"Saved model at epoch {epoch+1}.")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if args.checkpoint_pretrain:
                torch.save(classifier.state_dict(), f"/scratch/user/baileyyeah/Alzheimer_my/results/{args.backbone}/best_model_{args.model_name}_bs{args.batch_size}.pth")
            else:
                torch.save(classifier.state_dict(), f"/scratch/user/baileyyeah/Alzheimer_my/results/{args.backbone}/scratch/best_model_{args.model_name}_bs{args.batch_size}.pth")
            print(f"New best model saved at epoch {epoch+1} with val_acc: {val_acc:.4f}")
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_avg_loss:.4f}  Val Acc: {val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SwinViT Classifier on ADNI MRI dataset")

    parser.add_argument("--train_json", type=str, required=True, help="Path to training JSON file")
    parser.add_argument("--train_image_dir", type=str, required=True, help="Path to training image directory")
    parser.add_argument("--valid_json", type=str, required=True, help="Path to validation JSON file")
    parser.add_argument("--valid_image_dir", type=str, required=True, help="Path to validation image directory")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--backbone", type=str, default='resnet')
    parser.add_argument("--checkpoint_pretrain", type=str, default=None)
    parser.add_argument("--model_name", type=str, default='resnet10')
    args = parser.parse_args()
    print(args)
    train(args)
