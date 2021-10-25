from sklearn.metrics import roc_auc_score
from utils import *
import time

criterion = torch.nn.functional.cross_entropy
# criterion = torch.nn.MultiLabelSoftMarginLoss


def search(model, train_loader, val_loader, train_mask, val_mask, args, logger):
    start = time.time()

    device = get_device(args)
    model.to(device)
    # if args.task == 'graph':
    #     data_size = len(dataset)
    #     train_loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=args.bs, shuffle=True)
    #     val_loader = DataLoader(dataset[int(data_size * 0.8):int(data_size * 0.9)], batch_size=args.bs, shuffle=True)
    #     test_loader = DataLoader(dataset[int(data_size * 0.9):], batch_size=args.bs, shuffle=True)
    # else:
    #     trian_loader = val_loader = test_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True)
    # # train_loader, val_loader, test_loader = dataloaders
    optimizer = get_optimizer(model, args)
    # for name, param in model.named_parameters():
    #     print(name, '\t\t', param.shape)

    metric = args.metric
    recorder = SearchRecorder(metric)
    # train_mask, val_mask, test_mask = reset_mask(dataset, args, logger, stratify=None)

    for step in range(args.epoch):
        # print(model.log_alpha_agg)
        # print(model.Z_agg_hard)
        # if step < 2:
        #     print('#########################################################################################')
        #     for n, p in model.named_parameters():
        #         print(n)
        #         print(p)

        optimize_model(model, train_loader, train_mask, optimizer, device, args)
        train_loss, train_acc, train_auc = eval_model(model, train_loader, train_mask, device, args, split='train')
        val_loss, val_acc, val_auc = eval_model(model, val_loader, val_mask, device, args, split='val')
        #         test_loss, test_acc, test_auc = eval_model(model, test_loader, device)
        #         recorder.update(train_acc, train_auc, val_acc, val_auc, test_acc, test_auc)
        #####################################################################################################
        #####################################################################################################
        model.update_z_hard()
        # if step > 30 and step % 5 == 0 and model.temperature >= 1e-20:
        if step > 10 and step % 2 == 0 and model.temperature >= 1e-20:
            model.temperature *= 1e-1
            # model.temperature /= 1.1
        #####################################################################################################
        #####################################################################################################

        recorder.update(train_acc, train_auc, val_acc, val_auc)

        logger.info('epoch %d best val %s: %.4f, train loss: %.4f; train %s: %.4f val %s: %.4f' %
                    (step, metric, recorder.get_best_metric()[0], train_loss,
                     metric, recorder.get_latest_metrics()[0],
                     metric, recorder.get_latest_metrics()[1]))
    logger.info('(Search Stage) best val acc: %.4f (epoch: %d)' % recorder.get_best_acc())
    logger.info('(Search Stage) best val auc: %.4f (epoch: %d)' % recorder.get_best_auc())

    results, max_step = recorder.get_best_metric()
    model.max_step = max_step
    model.best_metric_search = results

    end = time.time()
    search_time = end - start
    logger.info('Search time for {} : {} seconds' .format(args.dataset, search_time))

    return model, results


def retrain(model, train_loader, test_loader, train_mask, test_mask, args, logger):
    device = get_device(args)
    model.derive_arch()

    logger.info('Derived z')
    logger.info(model.searched_arch_z)
    logger.info('Derived arch')
    logger.info(model.searched_arch_op)

    def weight_reset(m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()
    model.apply(weight_reset)
    model.to(device)

    # train_loader, val_loader, test_loader = dataloaders
    optimizer = get_optimizer(model, args)
    metric = args.metric
    #     recorder = Recorder(metric)
    recorder = RetrainRecorder(metric)
    for step in range(args.retrain_epoch):
        optimize_model(model, train_loader, train_mask, optimizer, device, args)
        train_loss, train_acc, train_auc = eval_model(model, train_loader, train_mask, device, args, split='train')
        # test_loss, test_acc, test_auc = eval_model(model, test_loader, test_mask, device, args, split='val')
        test_loss, test_acc, test_auc = eval_model(model, test_loader, test_mask, device, args, split='test')

        #         recorder.update(train_acc, train_auc, val_acc, val_auc, test_acc, test_auc)
        #         recorder.update(train_acc, train_auc, val_acc, val_auc)
        recorder.update(train_acc, train_auc, test_acc, test_auc)

        logger.info('epoch %d best test %s: %.4f, retrain loss: %.4f; retrain %s: %.4f test %s: %.4f' %
                    (step, metric, recorder.get_best_metric()[0], train_loss,
                     metric, recorder.get_latest_metrics()[0],
                     metric, recorder.get_latest_metrics()[1]))
    logger.info('(Retrain Stage) best test acc: %.4f (epoch: %d)' % recorder.get_best_acc())
    logger.info('(Retrain Stage) best test auc: %.4f (epoch: %d)' % recorder.get_best_auc())

    results, max_step = recorder.get_best_metric()
    model.max_step = max_step
    model.best_metric_retrain = results
    return model, results


def optimize_model(model, dataloader, mask, optimizer, device, args):
    model.train()
    # setting of data shuffling move to dataloader creation
    for batch in dataloader:
        batch = batch.to(device)
        label = batch.y
        prediction = model(batch)
        # if args.task == 'node':
        if args.task == 'node' and args.dataset != 'PPI':
            # label = label[batch.train_mask]
            # prediction = prediction[batch.train_mask]
            label = label[mask]
            prediction = prediction[mask]

        # print(prediction)
        # print(label)
        # print(prediction.shape)
        # print(label.shape)

        loss = criterion(prediction, label, reduction='mean')
        # loss.backward()
        loss.backward(retain_graph=True)
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()


def eval_model(model, dataloader, mask, device, args, split, return_predictions=False):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            label = batch.y
            prediction = model(batch)
            # if args.task == 'node':
            if args.task == 'node' and args.dataset != 'PPI':
                # if split == 'train':
                #     mask = batch.train_mask
                # elif split == 'valid':
                #     mask = batch.val_mask
                # else:
                #     mask = batch.test_mask
                # mask = data.val_mask if is_validation else data.test_mask
                # node classification: only evaluate on nodes in test set
                prediction = prediction[mask]
                label = label[mask]
            labels.append(label)
            predictions.append(prediction)
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
    if not return_predictions:
        loss, acc, auc = compute_metric(predictions, labels)
        return loss, acc, auc
    else:
        return predictions


def compute_metric(predictions, labels):
    with torch.no_grad():
        # compute loss:
        loss = criterion(predictions, labels, reduction='mean').item()
        # compute acc:
        correct_predictions = (torch.argmax(predictions, dim=1) == labels)
        acc = correct_predictions.sum().cpu().item()/labels.shape[0]
        # compute auc:
        predictions = torch.nn.functional.softmax(predictions, dim=-1)
        multi_class = 'ovr'
        if predictions.size(1) == 2:
            predictions = predictions[:, 1]
            multi_class = 'raise'
        auc = roc_auc_score(labels.cpu().numpy(), predictions.cpu().numpy(), multi_class=multi_class)
    return loss, acc, auc


class SearchRecorder:
    """
    always return test numbers except the last method
    """

    def __init__(self, metric):
        self.metric = metric
        self.train_accs, self.val_accs, self.train_aucs, self.val_aucs = [], [], [], []

    def update(self, train_acc, train_auc, val_acc, val_auc):
        self.train_accs.append(train_acc)
        self.train_aucs.append(train_auc)
        self.val_accs.append(val_acc)
        #         self.test_accs.append(test_acc)
        self.val_aucs.append(val_auc)

    #         self.test_aucs.append(test_auc)

    def get_best_metric(self):
        dic = {'acc': self.get_best_acc(), 'auc': self.get_best_auc()}
        return dic[self.metric]

    def get_best_acc(self):
        #         if val:
        #             max_step = int(np.argmax(np.array(self.val_accs)))
        #         else:
        #             max_step = int(np.argmax(np.array(self.test_accs)))
        #         return self.test_accs[max_step], max_step
        max_step = int(np.argmax(np.array(self.val_accs)))
        return self.val_accs[max_step], max_step

    def get_best_auc(self):
        #         if val:
        #             max_step = int(np.argmax(np.array(self.val_aucs)))
        #         else:
        #             max_step = int(np.argmax(np.array(self.test_aucs)))
        #         return self.test_aucs[max_step], max_step
        max_step = int(np.argmax(np.array(self.val_aucs)))
        return self.val_aucs[max_step], max_step

    def get_latest_metrics(self):
        if len(self.train_accs) < 0:
            raise Exception
        if self.metric == 'acc':
            return self.train_accs[-1], self.val_accs[-1]
        elif self.metric == 'auc':
            return self.train_aucs[-1], self.val_aucs[-1]
        else:
            raise NotImplementedError


#     def get_best_val_metric(self):
#         max_step = self.get_best_auc()[1]
#         dic = {'acc': (self.val_accs[max_step], max_step), 'auc': (self.val_aucs[max_step], max_step)}
#         return dic[self.metric]


class RetrainRecorder:
    """
    always return test numbers except the last method
    """

    def __init__(self, metric):
        self.metric = metric
        self.train_accs, self.test_accs, self.train_aucs, self.test_aucs = [], [], [], []

    def update(self, train_acc, train_auc, test_acc, test_auc):
        self.train_accs.append(train_acc)
        self.train_aucs.append(train_auc)
        #         self.val_accs.append(val_acc)
        self.test_accs.append(test_acc)
        #         self.val_aucs.append(val_auc)
        self.test_aucs.append(test_auc)

    def get_best_metric(self):
        dic = {'acc': self.get_best_acc(), 'auc': self.get_best_auc()}
        return dic[self.metric]

    def get_best_acc(self):
        #         if val:
        #             max_step = int(np.argmax(np.array(self.val_accs)))
        #         else:
        #             max_step = int(np.argmax(np.array(self.test_accs)))
        #         return self.test_accs[max_step], max_step
        #         max_step = int(np.argmax(np.array(self.val_accs)))
        max_step = int(np.argmax(np.array(self.test_accs)))
        return self.test_accs[max_step], max_step

    def get_best_auc(self):
        #         if val:
        #             max_step = int(np.argmax(np.array(self.val_aucs)))
        #         else:
        #             max_step = int(np.argmax(np.array(self.test_aucs)))
        #         return self.test_aucs[max_step], max_step
        #         max_step = int(np.argmax(np.array(self.val_aucs)))
        max_step = int(np.argmax(np.array(self.test_aucs)))
        return self.test_aucs[max_step], max_step

    def get_latest_metrics(self):
        if len(self.train_accs) < 0:
            raise Exception
        if self.metric == 'acc':
            return self.train_accs[-1], self.test_accs[-1]
        elif self.metric == 'auc':
            return self.train_aucs[-1], self.test_aucs[-1]
        else:
            raise NotImplementedError

#     def get_best_val_metric(self):
#         max_step = self.get_best_auc()[1]
#         dic = {'acc': (self.val_accs[max_step], max_step), 'auc': (self.val_aucs[max_step], max_step)}
#         return dic[self.metric]