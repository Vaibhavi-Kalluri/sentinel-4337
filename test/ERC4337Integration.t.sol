// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";

import {EntryPoint} from "account-abstraction/core/EntryPoint.sol";
import {SimpleAccountFactory} from "account-abstraction/accounts/SimpleAccountFactory.sol";
import {SimpleAccount} from "account-abstraction/accounts/SimpleAccount.sol";
import {PackedUserOperation} from "account-abstraction/interfaces/PackedUserOperation.sol";

import {SentinelTarget} from "../src/SentinelTarget.sol";

contract ERC4337IntegrationTest is Test {
    EntryPoint internal entryPoint;
    SimpleAccountFactory internal factory;
    SentinelTarget internal target;

    uint256 internal ownerPrivateKey;
    address internal owner;
    address internal beneficiary;

    function setUp() public {
        // Deploy fresh contracts inside this isolated Foundry test.
        entryPoint = new EntryPoint();
        factory = new SimpleAccountFactory(entryPoint);
        target = new SentinelTarget();

        ownerPrivateKey = 0xA11CE;
        owner = vm.addr(ownerPrivateKey);
        beneficiary = makeAddr("beneficiary");

        vm.txGasPrice(1 gwei);
    }

    function testSignedUserOperationExecutesThroughEntryPoint() public {
        uint256 salt = 1;

        // Calculate the counterfactual smart-account address.
        address sender = factory.getAddress(owner, salt);

        // The factory must be called through EntryPoint's SenderCreator.
        bytes memory initCode = abi.encodePacked(
            address(factory),
            abi.encodeCall(
                SimpleAccountFactory.createAccount,
                (owner, salt)
            )
        );

        // The account calls SentinelTarget.setValue(42).
        bytes memory innerCall = abi.encodeCall(
            SentinelTarget.setValue,
            (42)
        );

        bytes memory callData = abi.encodeWithSignature(
    "execute(address,uint256,bytes)",
    address(target),
    0,
    innerCall
);

        PackedUserOperation memory userOp;

        userOp.sender = sender;
        userOp.nonce = entryPoint.getNonce(sender, 0);
        userOp.initCode = initCode;
        userOp.callData = callData;

        // Pack verificationGasLimit and callGasLimit as two uint128 values.
        userOp.accountGasLimits = bytes32(
            (uint256(1_000_000) << 128) | uint256(300_000)
        );

        userOp.preVerificationGas = 50_000;

        // Pack maxPriorityFeePerGas and maxFeePerGas as two uint128 values.
        userOp.gasFees = bytes32(
            (uint256(1 gwei) << 128) | uint256(2 gwei)
        );

        userOp.paymasterAndData = "";

        // Prefund the counterfactual account's EntryPoint deposit.
        entryPoint.depositTo{value: 1 ether}(sender);

        // Sign the UserOperation hash with the account owner's private key.
        bytes32 userOpHash = entryPoint.getUserOpHash(userOp);

        (uint8 v, bytes32 r, bytes32 s) =
            vm.sign(ownerPrivateKey, userOpHash);

        userOp.signature = abi.encodePacked(r, s, v);

        PackedUserOperation[] memory ops =
            new PackedUserOperation[](1);
        ops[0] = userOp;

        // Submit through the real ERC-4337 EntryPoint.
vm.startPrank(owner, owner);
entryPoint.handleOps(ops, payable(beneficiary));
vm.stopPrank();        
        // Confirm account deployment and the intended target execution.
        assertGt(sender.code.length, 0);
        assertEq(target.value(), 42);
    }
}
