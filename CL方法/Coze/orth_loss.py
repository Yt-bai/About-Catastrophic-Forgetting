#     def _init_train(self, train_loader, test_loader, optimizer, scheduler):
#         prog_bar = tqdm(range(self.args['tuned_epoch']))
#         loss_cos = AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["m"])
#         for _, epoch in enumerate(prog_bar):
#             self._network.backbone.train()
#             losses = 0.0
#             correct, total = 0, 0
#             for i, (_, inputs, targets) in enumerate(train_loader):
#                 inputs, targets = inputs.to(self._device), targets.to(self._device)
#                 features = self._network.backbone(inputs, adapter_id=self._cur_task, train=True)["features"]
#                 logits = self._network.fc(features)["logits"]

#                 loss = loss_cos(logits[:, self._known_classes:], targets - self._known_classes)
#                 # loss = F.cross_entropy(logits, targets.long())
#                 if self._cur_task > 0 and self.use_orth:
#                     loss += self.orth_loss(features) * self.args["reg"] * torch.exp(-torch.tensor(self._cur_task+1, dtype=torch.float32, device=self._device))
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 losses += loss.item()

#                 _, preds = torch.max(logits, dim=1)
#                 correct += preds.eq(targets.expand_as(preds)).cpu().sum()
#                 total += len(targets)

#             if scheduler:
#                 scheduler.step()
#             train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

#             info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
#                 self._cur_task,
#                 epoch + 1,
#                 self.args['tuned_epoch'],
#                 losses / len(train_loader),
#                 train_acc,
#             )
#             prog_bar.set_description(info)

#         logging.info(info)
#     def orth_loss(self, features):
#         final_loss = 0

#         for i in range(self._cur_task):
#             loss = 0
#             for j in range(12):
#                 cur_up_proj = self._network.backbone.cur_adapter[j].up_proj.weight
#                 prev_up_proj = self._network.backbone.adapter_list[i][j].up_proj.weight
#                 cur_up_proj_normalized = F.normalize(cur_up_proj, p=2, dim=1)  # Normalize along the feature dimension
#                 prev_up_proj_normalized = F.normalize(prev_up_proj, p=2, dim=1)

#                 dot_product = torch.mean(torch.matmul(cur_up_proj_normalized, prev_up_proj_normalized.transpose(1, 0)))
#                 # loss += (torch.abs(dot_product)) / 12
#                 # cur_up_proj = self._network.backbone.cur_adapter[j].down_proj.weight
#                 # prev_up_proj = self._network.backbone.adapter_list[i][j].down_proj.weight
#                 # cur_up_proj_normalized = F.normalize(cur_up_proj, p=2, dim=1)  # Normalize along the feature dimension
#                 # prev_up_proj_normalized = F.normalize(prev_up_proj, p=2, dim=1)

#                 # dot_product = torch.mean(torch.matmul(cur_up_proj_normalized, prev_up_proj_normalized.transpose(1, 0)))


#                 loss += (torch.abs(dot_product)) / 12
#             final_loss += loss

#         return final_loss