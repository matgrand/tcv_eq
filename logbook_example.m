% Look for shots in shotlist and return a structure with:
% Run: date
% Shot: shot number
% Topic: we are only selecting PDJ entries, they usually log blips
% Text: the actual log (capped at 5000 characters)

shotlist = 86814:87118; % for small batches of shot u can do a single call
topic_list = "('PdJ')";
ddd = sprintf('SELECT Run, Shot, Username, Topic, Text, Entered FROM Entries WHERE (Shot in (%d%s)) AND (Topic in %s) and (voided is NULL) ORDER BY Shot DESC, Topic limit 5000',shotlist(1),sprintf(',%d',shotlist(2:end)),topic_list);
% F = sqllogbook(ddd);
% 
% % print detected blip numbers
% for i = 1 : numel(F.Text)
%   if contains(F.Text(i),'blip', 'IgnoreCase', true)
%     fprintf('Shot %d was a blip\n',F.Shot(i))
% 	end
% 	if contains(F.Text(i),'SN', 'IgnoreCase', true)
%     fprintf('Shot %d was a SN\n',F.Shot(i))
%   end
% end

for shot = shotlist
	fprintf('---------------------------------------------------------------')
	fprintf('shot %d\n', shot);
	ddd = sprintf('SELECT Run, Shot, Username, Topic, Text, Entered FROM Entries WHERE (Shot in (%d%s)) AND (Topic in %s) and (voided is NULL) ORDER BY Shot DESC, Topic limit 5000',shot,sprintf(',%d',shot),topic_list);
	F = sqllogbook(ddd);
	try
		% print detected blip numbers
		fprintf('Text: %s \n', F.Text{:})
		for i = 1 : numel(F.Text)
			if contains(F.Text(i),'blip', 'IgnoreCase', true)
				fprintf('Shot %d was a blip\n',F.Shot(i))
				pause(0.5)
			end
			if contains(F.Text(i),'DEAB', 'IgnoreCase', true)
				fprintf('Shot %d was a DEAB\n',F.Shot(i))
				pause(0.5)
			end
		end
	catch ME
		fprintf('Error processing shot %d: %s \n', shot, ME.message);
	end
% 	pause(0.1)
end

















